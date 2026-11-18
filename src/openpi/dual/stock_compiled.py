"""Inference adapter preserving stock OpenPI prefix/expert tensor operations.

No foundation weights, attention arithmetic, masks or token layout are changed.
Only the sampling boundary and KV values are injectable. Compilation is explicit.
"""
from dataclasses import dataclass
from functools import cached_property
import torch
from transformers.cache_utils import DynamicCache
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.dual.bridge import FrozenPi05, PrefixKV

def stock_prefix(model, observation):
    # This is the prefix portion of the unmodified PI0Pytorch.sample_actions.
    images, image_masks, tokens, text_mask, state = model._preprocess_observation(observation, train=False)
    return stock_prefix_tensors(model,images,image_masks,tokens,text_mask,state)

def stock_prefix_tensors(model,images,image_masks,tokens,text_mask,state):
    embs, mask, ar_mask = model.embed_prefix(images, image_masks, tokens, text_mask)
    attention = make_att_2d_masks(mask, ar_mask)
    positions = torch.cumsum(mask, dim=1) - 1
    attention = model._prepare_attention_masks_4d(attention)
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention, position_ids=positions, past_key_values=None,
        inputs_embeds=[embs, None], use_cache=True)
    return cache, mask, state

def stock_suffix(model, state, mask, cache, noise, num_steps=1, start_time=1.):
    # Same Euler arithmetic, dtype and time updates as stock sample_actions.
    # Static NFE replaces its tensor-valued while condition; verify exact NFE.
    dt = torch.tensor(-float(start_time)/num_steps, dtype=torch.float32, device=noise.device)
    time = torch.tensor(float(start_time), dtype=torch.float32, device=noise.device)
    x = noise
    for _ in range(num_steps):
        x = x + dt * model.denoise_step(state, mask, cache, x, time.expand(x.shape[0]))
        time = time + dt
    return x


@dataclass
class StockPrefix:
    cache: DynamicCache
    mask: torch.Tensor
    state: torch.Tensor

    # Compatibility for diagnostics only. Slow/expert never materialize stacks.
    @cached_property
    def k(self):
        return torch.stack([self.cache[i][0][:,0] for i in range(len(self.cache))],1)
    @cached_property
    def v(self):
        return torch.stack([self.cache[i][1][:,0] for i in range(len(self.cache))],1)

class StockCompiledBridge(FrozenPi05):
    """Native prefix/expert graphs with optional KV and action-latent adapters."""
    layout = "stock"
    fused_fast = True

    def __init__(self,checkpoint,device='cuda',text_tokens=32,compile_mode='max-autotune-no-cudagraphs'):
        super().__init__(checkpoint,device,text_tokens)
        if compile_mode not in {None,'default','max-autotune-no-cudagraphs'}:
            raise ValueError('CUDA graph output lifetimes are not validated')
        self.compile_mode=compile_mode
        # Reuse a single stable stock function/code object. Compiling wrapper
        # copies independently can choose different BF16 kernels.
        self._slow=self._compile(stock_prefix_tensors)
        self._expert=self._compile(stock_suffix)
        self._fast={};self._updater=None
        self.indices=torch.cat((torch.arange(512),torch.arange(768,768+text_tokens))).to(self.device)

    def _compile(self,fn,*,preserve_casts=False):
        if self.compile_mode is None:
            return fn
        if preserve_casts:
            options={**torch._inductor.list_mode_options(self.compile_mode),
                     'emulate_precision_casts':True,
                     'force_same_precision':True,
                     'max_autotune_gemm_backends':'ATEN',
                     'epilogue_fusion':False}
            return torch.compile(fn,fullgraph=True,options=options)
        return torch.compile(fn,fullgraph=True,mode=self.compile_mode)

    def hand(self,*args,**kwargs):
        raise RuntimeError('Stock inference must use expert(prefix, noise); raw hand would bypass compilation')

    def slow_tensor(self,*inputs):
        return stock_prefix_tensors(self.model,*inputs)

    @torch.no_grad()
    def prefix_observation(self,observation):
        inputs=self.model._preprocess_observation(observation,train=False)
        images,masks,tokens,text_mask,state=inputs
        if self._updater is not None and bool(text_mask[:,self.text_tokens:].any()):
            raise ValueError('Active text exceeds trained updater capacity')
        if len(images)!=3 or not bool(masks[0].all()) or not bool(masks[1].all()) or bool(masks[2].any()):
            raise ValueError('Requires valid base/left wrist and masked right wrist')
        cache,mask,state=self._slow(self.model,*inputs)
        return StockPrefix(cache,mask,state)

    def cache(self,k,v):
        # Eager equivalence diagnostic only; runtime preserves native cache.
        dtype=self.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.k_proj.weight.dtype
        return DynamicCache.from_legacy_cache(tuple((k[:,i,None].to(dtype),v[:,i,None].to(dtype)) for i in range(k.shape[1])))

    def expert_tensor(self,k,v,mask,state,noise,num_steps=1,start_time=1.):
        return stock_suffix(self.model,state,mask,self.cache(k,v),noise,num_steps,start_time)

    def updated_cache(self,updater,cache,mask,condition):
        # Gather only the updater's trained slots, without altering the original
        # full native KV layout or its masked values.
        k=torch.stack([cache[i][0][:,0].index_select(1,self.indices) for i in range(len(cache))],1)
        v=torch.stack([cache[i][1][:,0].index_select(1,self.indices) for i in range(len(cache))],1)
        compact_mask=mask.index_select(1,self.indices)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pk,pv=updater.forward_tensor(k,v,compact_mask,condition)
        def preserve_layout(original, values):
            updated=original.index_copy(2,self.indices,values.to(original.dtype))
            return torch.empty_strided(original.shape,original.stride(),dtype=original.dtype,
                                       device=original.device).copy_(updated)
        pairs=tuple((preserve_layout(cache[i][0],pk[:,i,None]),
                     preserve_layout(cache[i][1],pv[:,i,None])) for i in range(len(cache)))
        return DynamicCache.from_legacy_cache(pairs)

    def prepare(self,updater=None,num_steps=1,start_time=1.):
        if type(num_steps) is not int or num_steps<1 or not 0<start_time<=1:
            raise ValueError('Invalid integration schedule')
        if updater is not None:
            if self._updater is not None and self._updater is not updater:
                raise ValueError('One compiled bridge binds one frozen updater')
            if self._updater is None and (updater.training or any(p.requires_grad for p in updater.parameters())):
                raise ValueError('Compiled inference requires frozen eval updater')
            self._updater=updater
        key=num_steps,float(start_time)
        if updater is not None and 'updater' not in self._fast:
            def update(cache,mask,condition):
                return self.updated_cache(updater,cache,mask,condition)
            self._fast['updater']=self._compile(update,preserve_casts=True)

    def expert(self,prefix,noise,num_steps=1,start_time=1.):
        self.prepare(num_steps=num_steps,start_time=start_time)
        return self._expert(self.model,prefix.state,prefix.mask,prefix.cache,noise,num_steps,start_time)

    def fast(self,updater,prefix,condition,noise,num_steps=1,start_time=1.):
        if (condition.shape!=(prefix.mask.shape[0],updater.config.condition_dim)
                or condition.device!=prefix.mask.device):
            raise ValueError('Invalid updater condition shape/device')
        if not bool(torch.isfinite(condition).all()) or bool(((condition[:,0]<0)|(condition[:,0]>1)).any()):
            raise ValueError('Updater condition must be finite with age in [0,1]')
        if not getattr(prefix,'updater_layout_validated',False):
            if bool(prefix.mask[:,512:768].any()) or bool(prefix.mask[:,768+self.text_tokens:].any()):
                raise ValueError('Active prefix content lies outside trained updater slots')
            prefix.updater_layout_validated=True
        self.prepare(updater,num_steps,start_time)
        # Two compiled graphs, one stream, no intervening synchronization:
        # preserve the original expert executable instead of fusing a copy.
        predicted=self._fast['updater'](prefix.cache,prefix.mask,condition)
        return self._expert(self.model,prefix.state,prefix.mask,predicted,noise,num_steps,start_time)

    @torch.no_grad()
    def warmup(self,prefix,updater,config):
        """Compile configured branches; publish no snapshot/history/endpoint."""
        from openpi.dual.rose_tfws import integration_schedule
        noise=torch.zeros((1,10,32),device=self.device)
        schedules={(config.num_steps,1.)}
        if config.cache.enabled:
            start=config.cache.restart_time
            nfe,_=integration_schedule(config.num_steps,start)
            schedules.add((config.cache.warm_nfe or nfe,start))
        condition=torch.zeros((1,config.updater.condition_dim),device=self.device)
        condition[:,0]=1./config.updater.history
        for nfe,start in sorted(schedules):
            for _ in range(2):
                self.expert(prefix,noise,nfe,start)
                if updater is not None:self.fast(updater,prefix,condition,noise,nfe,start)
        torch.cuda.current_stream(self.device).synchronize()

