"""Quota-aware storage checks for ROSE; never delete scientific artifacts."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil
import time

GIB = 1024**3


def snapshot(path):
    target = Path(path).resolve()
    while not target.exists():
        target = target.parent
    root = Path(os.environ.get('PI_FS_STORAGE_ROOT', target)).resolve()
    if 'PI_FS_STORAGE_ROOT' in os.environ and target != root and root not in target.parents:
        raise ValueError(f'Write target {target} is outside storage root {root}')
    free = shutil.disk_usage(target).free
    quotas = []
    for directory in (root, *root.parents):
        try:
            limit = int(os.getxattr(directory, 'ceph.quota.max_bytes'))
            if not limit:
                continue
            used = int(os.getxattr(directory, 'ceph.dir.rbytes'))
        except OSError as error:
            if error.errno not in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
            continue
        quotas.append({'root': str(directory), 'limit_bytes': limit, 'used_bytes': used})
        free = min(free, max(0, limit - used))
    if os.environ.get('PI_FS_REQUIRE_CEPH_QUOTA') == '1' and not quotas:
        raise RuntimeError(f'Expected readable Ceph quota for {root}; refusing filesystem-only check')
    return {'available_bytes': free, 'quotas': quotas}


def metrics(path):
    state = snapshot(path)
    result = {'storage/available_gib': state['available_bytes'] / GIB}
    if state['quotas']:
        limiting = min(state['quotas'], key=lambda q: q['limit_bytes'] - q['used_bytes'])
        result.update({'storage/quota_gib': limiting['limit_bytes'] / GIB,
                       'storage/used_gib': limiting['used_bytes'] / GIB})
    return result


def wait_for_space(path, required_bytes=0, *, event_path=None, on_event=None):
    """Wait before work/writes, preserving committed data and the live model.

    Required bytes are additional to the explicit safety reserve. Ceph directory
    accounting is asynchronous, so concurrent writers need a conservative reserve.
    No checkpoint, dataset or cache is automatically deleted by this function.
    """
    reserve = int(float(os.environ.get('PI_FS_STORAGE_MIN_FREE_GIB', '1')) * GIB)
    threshold = int(required_bytes) + reserve
    paused = False
    while True:
        state = snapshot(path)
        enough = state['available_bytes'] >= threshold
        if enough and not paused:
            return state
        event = {'event': 'storage_resumed' if enough else 'storage_wait',
                 'time_unix': time.time(), 'required_bytes': int(required_bytes),
                 'reserve_bytes': reserve, **state}
        print(json.dumps(event), flush=True)
        if event_path:
            with Path(event_path).open('a') as stream:
                stream.write(json.dumps(event) + '\n')
        if on_event:
            on_event({'storage/waiting': int(not enough),
                      'storage/available_gib': state['available_bytes'] / GIB})
        if enough:
            return state
        paused = True
        time.sleep(30)
