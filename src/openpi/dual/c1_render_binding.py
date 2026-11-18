"""Read the physical UUID from the GL context actually used by robosuite."""
import ctypes
import os
import uuid


def _uuid(value):
    return str(uuid.UUID(str(value).removeprefix('GPU-')))


def verify_render_device(env):
    import torch
    from OpenGL import GL
    from OpenGL.raw.GL.EXT.memory_object import GL_DEVICE_UUID_EXT, glGetUnsignedBytei_vEXT

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError(f'Expected paired CUDA-visible GPUs, received {visible}')
    context = env.env.sim._render_context_offscreen
    context.gl_ctx.make_current()
    raw = (ctypes.c_ubyte * 16)()
    glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, 0, raw)
    gl_uuid = str(uuid.UUID(bytes=bytes(raw)))
    inference = torch.cuda.get_device_properties(0)
    renderer = torch.cuda.get_device_properties(1)
    if gl_uuid != _uuid(renderer.uuid) or gl_uuid == _uuid(inference.uuid):
        raise RuntimeError(f'Expected GL on cuda:1 ({renderer.uuid}), got {gl_uuid}; policy is on {inference.uuid}')
    return dict(inference_gpu_uuid=str(inference.uuid), render_gpu_uuid=str(renderer.uuid),
                inference_gpu_name=inference.name, render_gpu_name=renderer.name,
                gl_device_uuid=gl_uuid, gl_renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                egl_device_id=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
                cuda_visible_devices=visible, cpu_affinity=sorted(os.sched_getaffinity(0)),
                torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                verification='glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT,0) on active LIBERO render context')
