"""Cache compatibility, failed publication and cross-process tuning exclusion."""
import json
import multiprocessing

import pytest

from mxfp6 import _autotune_cache as cache
from mxfp6._autotune_policy import AutotuneResult, W6A8Config


def result(config_id=2):
    return AutotuneResult(W6A8Config(config_id, 1, 0), 1.0, 2.0, 1.5, 3)


def test_legacy_entry_and_descriptor_identity(tmp_path):
    descriptor = {"schema": 1, "problem": {"m": 17, "n": 136, "k": 128}}
    key = cache.cache_key(descriptor)
    assert key == cache.cache_key(dict(reversed(list(descriptor.items()))))
    path = tmp_path / f"{key}.json"
    path.write_text(json.dumps({"descriptor": descriptor,
        "config": {"config_id": 2, "swizzle": 1, "raster_order": 0,
                   "kernel": "legacy informational name"}}))
    assert cache.read_entry(tmp_path, key, descriptor) == result().config
    assert cache.read_entry(tmp_path, key, {**descriptor, "schema": 2}) is None
    cache.write_entry(tmp_path, key, descriptor, result(-1))
    assert cache.read_entry(tmp_path, key, descriptor) == result(-1).config


@pytest.mark.parametrize("payload", ['{', '[]', '{"descriptor": {}}',
    '{"descriptor": {}, "config": {"config_id": 999, "swizzle": 1, "raster_order": 0}}'])
def test_bad_entries_are_misses(tmp_path, payload):
    (tmp_path / 'key.json').write_text(payload)
    assert cache.read_entry(tmp_path, 'key', {}) is None


def test_failed_publication_keeps_previous_entry(tmp_path, monkeypatch):
    cache.write_entry(tmp_path, 'key', {}, result())
    def fail(*args):
        raise OSError('publication failed')
    monkeypatch.setattr(cache.os, 'replace', fail)
    with pytest.raises(OSError, match='publication failed'):
        cache.write_entry(tmp_path, 'key', {}, result(3))
    assert cache.read_entry(tmp_path, 'key', {}) == result().config
    assert not list(tmp_path.glob('*.tmp'))


def _cache_worker(root, ready, start, outcomes):
    ready.put(True)
    start.wait(10)
    with cache.file_lock(root, 'key'):
        hit = cache.read_entry(root, 'key', {})
        if hit is None:
            cache.write_entry(root, 'key', {}, result())
        outcomes.put(hit is None)


def test_processes_only_publish_one_decision(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    ready, outcomes, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_cache_worker, args=(tmp_path, ready, start, outcomes))
               for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=30)
        start.set()
        assert sorted(outcomes.get(timeout=30) for _ in workers) == [False, True]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        ready.close()
        outcomes.close()
