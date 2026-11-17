import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build_resources',ROOT/'scripts/build_resources.py')
resources=importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def test_parallelism_is_bounded_by_cpu_and_memory():
    gib=resources.GIB
    assert resources.choose_jobs(128,512*gib)==8
    assert resources.choose_jobs(128,64*gib)==2
    assert resources.choose_jobs(2,512*gib)==2
    assert resources.choose_jobs(128,16*gib)==1
    assert resources.choose_jobs(1,0)==1
