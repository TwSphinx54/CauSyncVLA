"""python -m openpi.robot.server --config /path/to/config.json"""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
from openpi.dual.config import config_from_dict
from openpi.robot.session import PROTOCOL, SessionPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class SingleRobotServer(WebsocketPolicyServer):
    # Keep OpenPI's message encoding/health check unchanged; own one physical session.
    active_connection = None

    async def _handler(self, websocket):
        if self.active_connection is not None:
            await websocket.close(code=1013, reason='This server already controls one robot session')
            return
        self.active_connection = websocket
        try:
            await super()._handler(websocket)
        finally:
            self._policy.reset()
            self.active_connection = None


def read_config(path, base=None, updater=None, eager=False):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    system = raw['system']
    def resolve(value):
        p = Path(value).expanduser()
        return str(p.resolve() if p.is_absolute() else (path.parent / p).resolve())
    system['base_checkpoint'] = str(Path(base).expanduser().resolve()) if base else resolve(system['base_checkpoint'])
    if updater:
        system['dual']['updater_checkpoint'] = str(Path(updater).expanduser().resolve())
    elif system['dual'].get('updater_checkpoint'):
        system['dual']['updater_checkpoint'] = resolve(system['dual']['updater_checkpoint'])
    if eager:
        system['inference']['compile_mode'] = None
    return config_from_dict(system), raw['deployment']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--base', help='Relocated converted PyTorch checkpoint directory')
    parser.add_argument('--updater', help='FR3 updater .pt; never the LIBERO updater')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--eager', action='store_true', help='Debug only; disable compilation')
    parser.add_argument('--print-config', action='store_true', help='Resolve paths without loading models')
    args = parser.parse_args()
    config, deployment = read_config(args.config, args.base, args.updater, args.eager)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return
    memory_budget_mib = int(os.environ.get('OPENPI_GPU_MEMORY_BUDGET_MIB', '0'))
    if memory_budget_mib:
        import torch
        total = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(memory_budget_mib * 1024**2 / total, args.device)
    from openpi.robot.live_policy import create_robot_policy
    logging.basicConfig(level=logging.INFO)
    from openpi.robot.execution_modes import execution_profile
    mode = deployment.get('execution_mode', 'accelerated100')
    profile = execution_profile(mode)
    policy = create_robot_policy(config, source_base_checkpoint=deployment['source_base_checkpoint'], device=args.device)
    metadata = dict(protocol=PROTOCOL, config=config.to_dict(),
        torch_allocator_budget_mib=memory_budget_mib or None,
        action_horizon=10, replan_steps=config.replan_steps, actions_per_response=1,
        state='absolute xyz / SciPy lowercase xyz Euler (extrinsic) / previous commanded gripper; 1=open',
        slow_schedule='ASAP latest observed frame; no fixed-rate gate',
        recorded_data_hz=10)
    from openpi.robot.async_session import AsyncSessionPolicy
    metadata['compatible_feedback10']=False
    metadata['pose_gripper_reanchor_v2']=True
    metadata['accelerated_feedback100']=mode=='accelerated100'
    metadata['accelerated_feedback200']=mode=='accelerated200'
    metadata['execution_mode']=mode
    metadata['host_dispatch_hz_limit']=profile.dispatch_hz
    metadata['model_feedback_hz_limit']=profile.feedback_hz
    metadata['trained_action_node_ms']=deployment.get('trained_action_node_ms',100)
    metadata['model_version']=deployment.get('model_version','feedback100_20261114' if config.dual.enabled else 'pi05_fr3')
    metadata['training_data_hz']=deployment.get('training_data_hz',10)
    session = AsyncSessionPolicy(policy, metadata)
    try:
        SingleRobotServer(session, host=args.host, port=args.port, metadata=metadata).serve_forever()
    finally:
        policy.close()


if __name__ == '__main__':
    main()
