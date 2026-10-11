"""Fixed BF16 BA projection: public wrapper, native contract and graph replay."""

import pytest
import torch

import mxfp6.mxfp8 as api


def test_public_wrapper_loads_existing_extension(monkeypatch):
    calls = []
    x, weight, output = object(), object(), object()
    monkeypatch.setattr(api, "load_library", lambda: calls.append("load"))

    def run(actual_x, actual_weight):
        assert actual_x is x and actual_weight is weight
        calls.append("run")
        return output

    monkeypatch.setattr(torch.ops.mxfp8_sm120, "bf16_ba_m16", run, raising=False)
    assert api.bf16_ba_m16(x, weight) is output
    assert calls == ["load", "run"]
    assert "bf16_ba_m16" in api.__all__


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("SM120 required")
    api.load_library()
    return device


def _inputs(device):
    return (
        torch.randn(16, 2048, device=device, dtype=torch.bfloat16),
        torch.randn(32, 2048, device=device, dtype=torch.bfloat16),
    )


def _assert_bytes(actual, expected):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_native_schema_and_exact_bounded_arithmetic(cuda_device):
    assert str(torch.ops.mxfp8_sm120.bf16_ba_m16.default._schema) == (
        "mxfp8_sm120::bf16_ba_m16(Tensor x, Tensor weight) -> Tensor"
    )
    # All partial sums are exactly representable in FP32, so this independent
    # FP64 reference checks the complete dot products and output coordinates.
    x = torch.randint(-2, 3, (16, 2048), device=cuda_device).to(torch.bfloat16)
    weight = torch.randint(-2, 3, (32, 2048), device=cuda_device).to(torch.bfloat16)
    expected = (x.double() @ weight.double().T).to(torch.bfloat16)
    output = api.bf16_ba_m16(x, weight)
    assert output.shape == (16, 32)
    assert output.dtype == torch.bfloat16 and output.device == x.device
    _assert_bytes(output, expected)


def test_changing_inputs_graph_on_current_stream(cuda_device):
    x, weight = _inputs(cuda_device)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(torch.cuda.current_stream(cuda_device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        api.bf16_ba_m16(x, weight)
        with torch.cuda.graph(graph, stream=stream):
            output = api.bf16_ba_m16(x, weight)
        previous = None
        for seed in (21, 22, 23, 24):
            torch.manual_seed(seed)
            new_x, new_weight = _inputs(cuda_device)
            x.copy_(new_x)
            weight.copy_(new_weight)
            expected = api.bf16_ba_m16(x, weight)
            graph.replay()
            _assert_bytes(output, expected)
            if previous is not None:
                assert not torch.equal(output, previous)
            previous = output.clone()
    stream.synchronize()


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("invalid", ["dtype", "shape", "rank", "contiguous", "alignment", "cpu"])
def test_invalid_native_inputs(cuda_device, operand, invalid):
    values = list(_inputs(cuda_device))
    rows = values[operand].size(0)
    messages = {
        "dtype": "BF16", "shape": "X\\[16,2048\\]", "rank": "X\\[16,2048\\]",
        "contiguous": "contiguous", "alignment": "16-byte-aligned", "cpu": "CUDA",
    }
    if invalid == "dtype":
        values[operand] = values[operand].float()
    elif invalid == "shape":
        values[operand] = values[operand][:-1]
    elif invalid == "rank":
        values[operand] = values[operand].reshape(-1)
    elif invalid == "contiguous":
        values[operand] = torch.empty(2048, rows, device=cuda_device, dtype=torch.bfloat16).T
    elif invalid == "alignment":
        storage = torch.empty(rows * 2048 + 1, device=cuda_device, dtype=torch.bfloat16)
        values[operand] = storage[1:].view(rows, 2048)
        assert values[operand].is_contiguous() and values[operand].data_ptr() % 16 != 0
    else:
        values[operand] = values[operand].cpu()
    with pytest.raises((RuntimeError, NotImplementedError), match=messages[invalid]):
        torch.ops.mxfp8_sm120.bf16_ba_m16(*values)


def test_mismatched_cuda_devices(cuda_device):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    x, weight = _inputs(cuda_device)
    other = (cuda_device + 1) % torch.cuda.device_count()
    with pytest.raises(RuntimeError, match="one CUDA device"):
        torch.ops.mxfp8_sm120.bf16_ba_m16(x, weight.to(other))


def test_device_guard_uses_input_device(cuda_device):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    x, weight = _inputs(cuda_device)
    expected = api.bf16_ba_m16(x, weight)
    other = (cuda_device + 1) % torch.cuda.device_count()
    with torch.cuda.device(other):
        actual = api.bf16_ba_m16(x, weight)
        assert torch.cuda.current_device() == other
    _assert_bytes(actual, expected)
