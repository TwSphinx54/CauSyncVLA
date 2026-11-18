"""Separate an immutable rollout protocol from scheduler execution attempts."""
import json
from datetime import datetime, timezone
from pathlib import Path


def resume_manifest(path, candidate, reviewed_source=None):
    path = Path(path)
    if not path.exists():
        if reviewed_source:
            raise ValueError('Reviewed source migration requires an existing manifest')
        return candidate
    prior = json.loads(path.read_text())
    old, new = dict(prior), dict(candidate)
    old.pop('job_id', None)
    new.pop('job_id', None)
    if reviewed_source:
        if prior.get('source_path') != reviewed_source:
            raise ValueError('Reviewed resume source does not match original manifest')
        # Explicit source migration changes only execution bookkeeping/storage.
        # All model/data/seed/runtime fields below must still match exactly.
        old.pop('source_path', None)
        new.pop('source_path', None)
        if (old.get('identity_mode') == new.get('identity_mode') == 'path_config_no_hash_verification'
                and all(v is None for v in old.get('source_sha256', {}).values())
                and all(v is None for v in new.get('source_sha256', {}).values())):
            old.pop('source_sha256', None)
            new.pop('source_sha256', None)
    if old != new:
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError(f'Resume protocol differs in {changed}; use a new output directory')
    return prior


def record_attempt(out, candidate, reviewed_source=None):
    attempt = {'utc': datetime.now(timezone.utc).isoformat(), 'job_id': candidate.get('job_id'),
               'source_path': candidate.get('source_path'), 'reviewed_previous_source': reviewed_source}
    with (Path(out) / 'execution_attempts.jsonl').open('a') as stream:
        stream.write(json.dumps(attempt) + '\n')
    return attempt
