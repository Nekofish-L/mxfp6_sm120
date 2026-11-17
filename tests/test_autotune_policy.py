"""Search ordering and refinement contracts, runnable without a GPU."""
import pytest

from mxfp6._autotune_policy import W6A8Config, candidate_kernel_ids, refinement_kernel_ids


@pytest.mark.parametrize("stream_k", [True, False])
@pytest.mark.parametrize("k", [128, 1023, 1024, 5120])
def test_default_candidate_boundaries(stream_k, k):
    def expected(ids):
        return tuple(i for i in ids if (stream_k and k >= 1024) or i not in (3,16))
    for m in (1,16,17,32,33,128,512):
        assert candidate_kernel_ids(m,k,all_families_max_m=512,stream_k_enabled=stream_k) == expected(range(30))
    assert candidate_kernel_ids(513,k,all_families_max_m=512,stream_k_enabled=stream_k) == expected(range(6,17))


def test_custom_threshold_preserves_legacy_shortlists():
    assert candidate_kernel_ids(16,2048,all_families_max_m=0,stream_k_enabled=True) == (0,1,2,3,4,5,29)
    assert candidate_kernel_ids(32,2048,all_families_max_m=0,stream_k_enabled=True) == (
        4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28)
    assert candidate_kernel_ids(33,2048,all_families_max_m=0,stream_k_enabled=True) == tuple(range(6,17))


def test_refinement_uses_best_swizzle_and_stable_ties():
    coarse = [(8.,W6A8Config(9,1,0)), (3.,W6A8Config(7,1,0)),
              (3.,W6A8Config(9,4,0)), (2.,W6A8Config(4,1,0)),
              (9.,W6A8Config(2,1,0))]
    # ID 9 was inserted before ID 7: retain that order for equal minima.
    assert refinement_kernel_ids(coarse,[9,7,4,2],exhaustive=False) == [4,9,7]
    assert refinement_kernel_ids(coarse,[9,7,4,2],exhaustive=True) == [4,9,7,2]
    assert refinement_kernel_ids([], [9,7,4,2],exhaustive=True) == [9,7,4]
