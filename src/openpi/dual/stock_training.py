"""Differentiable stock-layout adapter; foundation modules stay frozen.

Only active slots are stored for the updater. The expert receives the original
968-slot layout. Masked slots may be zero; none can contribute to attention.
"""
import torch
from transformers.cache_utils import DynamicCache
from openpi.dual.bridge import FrozenPi05, PrefixKV
from openpi.dual.stock_compiled import stock_prefix, stock_suffix

class StockTrainingBridge(FrozenPi05):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.indices=torch.cat((torch.arange(512),torch.arange(768,768+self.text_tokens))).to(self.device)

    @torch.no_grad()
    def prefix_observation(self,observation):
        cache,mask,state=stock_prefix(self.model,observation)
        if bool(mask[:,768+self.text_tokens:].any()):
            raise ValueError('Active text exceeds updater capacity')
        k=torch.stack([cache[i][0][:,0].index_select(1,self.indices) for i in range(len(cache))],1)
        v=torch.stack([cache[i][1][:,0].index_select(1,self.indices) for i in range(len(cache))],1)
        return PrefixKV(k,v,mask.index_select(1,self.indices),state)

    def hand(self,k,v,mask,state,noise,num_steps=1,start_time=1.):
        dtype=self.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.k_proj.weight.dtype
        def full(x):
            return x.new_zeros((x.shape[0],x.shape[1],968,x.shape[3])).index_copy(2,self.indices,x).to(dtype)
        fk,fv=full(k),full(v)
        fm=mask.new_zeros((mask.shape[0],968)).index_copy(1,self.indices,mask)
        cache=DynamicCache.from_legacy_cache(tuple((fk[:,i,None],fv[:,i,None]) for i in range(k.shape[1])))
        return stock_suffix(self.model,state,fm,cache,noise,num_steps,start_time)
