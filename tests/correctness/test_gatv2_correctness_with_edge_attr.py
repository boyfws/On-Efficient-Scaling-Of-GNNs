"""Correctness tests for fused CUDA GATv2 with edge attributes.

The CUDA kernel consumes already projected edge attributes with shape [E, H, D]
and computes, for every COO edge ``src -> dst``::

    z = x_left[dst] + x_right[src] + edge_attr[e]
    score = attn * LeakyReLU(z)
    out[dst] = sum(softmax(score) * x_right[src])

The reference below is deliberately functional rather than a second Conv class:
it isolates the fused aggregation kernel and lets us compare every input gradient.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets
from turbo_gnn.ops import gatv2_aggr


#pytestmark = pytest.mark.cuda


def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Stable softmax over incoming edges of every destination node."""
    num_heads = scores.size(1)
    dst_expanded = dst[:, None].expand(-1, num_heads)

    max_per_dst = torch.full(
        (num_nodes, num_heads),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    max_per_dst.scatter_reduce_(
        0,
        dst_expanded,
        scores,
        reduce="amax",
        include_self=True,
    )

    exp_scores = torch.exp(scores - max_per_dst[dst])
    denominator = torch.zeros_like(max_per_dst)
    denominator.scatter_add_(0, dst_expanded, exp_scores)
    return exp_scores / denominator[dst].clamp_min(1e-16)


def gatv2_edge_attr_reference(
    edge_index: torch.Tensor,
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    attention_weights: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    num_nodes: int,
    negative_slope: float,
) -> torch.Tensor:
    """Pure-PyTorch GATv2 aggregation with projected edge attributes."""
    src, dst = edge_index

    pre_activation = x_left[dst] + x_right[src] + edge_attr
    activated = F.leaky_relu(pre_activation, negative_slope=negative_slope)
    scores = (activated * attention_weights.unsqueeze(0)).sum(dim=-1)
    alpha = _edge_softmax(scores, dst, num_nodes)

    messages = x_right[src] * alpha.unsqueeze(-1)
    out = torch.zeros(
        num_nodes,
        x_left.size(1),
        x_left.size(2),
        dtype=x_left.dtype,
        device=x_left.device,
    )
    out.scatter_add_(0, dst[:, None, None].expand_as(messages), messages)
    return out


def _make_directed_graph(
    num_nodes: int,
    num_random_edges: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create an unsorted, deduplicated directed COO graph.

    A directed ring guarantees at least one incoming edge for every node.  The
    final random permutation is important: it checks that CSR edge IDs map back
    to the original COO ordering used by ``edge_attr``.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    ring_src = torch.arange(num_nodes, device=device)
    ring_dst = (ring_src + 1) % num_nodes
    random_src = torch.randint(
        num_nodes,
        (num_random_edges,),
        device=device,
        generator=generator,
    )
    random_dst = torch.randint(
        num_nodes,
        (num_random_edges,),
        device=device,
        generator=generator,
    )

    src = torch.cat((ring_src, random_src))
    dst = torch.cat((ring_dst, random_dst))
    flat = torch.unique(src * num_nodes + dst)
    edge_index = torch.stack((flat // num_nodes, flat % num_nodes))

    permutation = torch.randperm(edge_index.size(1), device=device, generator=generator)
    return edge_index[:, permutation].contiguous()


def _build_cuda_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    quantile: float = -1,
) -> AdjacencyForwardBackwardWithNodeBuckets:
    """Build directed dual CSR and preserve original COO edge IDs."""
    return AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
        edge_index=edge_index,
        num_nodes=num_nodes,
        quantile=quantile,
        # Keep int64 for now.  In the current converter, CSR tensors are cast by
        # index_dtype, but forward/backward_edge_indices are not cast with them.
        index_dtype=None,
        is_directed=True,
        add_edge_attr=True,
    )


def _make_inputs(
    num_nodes: int,
    num_edges: int,
    heads: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        # Small values avoid heavily saturated softmax and make gradient
        # comparisons more diagnostic, especially in fp16/bf16.
        value = 0.2 * torch.randn(*shape, device=device, dtype=torch.float32, generator=generator)
        return value.to(dtype).contiguous().requires_grad_(requires_grad)

    return (
        randn(num_nodes, heads, head_dim),
        randn(num_nodes, heads, head_dim),
        randn(heads, head_dim),
        randn(num_edges, heads, head_dim),
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, name: str) -> None:
    difference = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        msg=(
            f"{name} mismatch: max|diff|={difference.max().item():.3e}, "
            f"mean|diff|={difference.mean().item():.3e}"
        ),
    )


def test_edge_ids_match_original_coo_order_after_csr_sort() -> None:
    """The two CSR permutations must point to the same original COO edges."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 5
    edge_index = torch.tensor(
        [
            [3, 0, 4, 1, 2, 0, 3],
            [1, 4, 0, 3, 1, 2, 4],
        ],
        device=device,
        dtype=torch.long,
    )
    graph = _build_cuda_graph(edge_index, num_nodes)

    forward_dst = torch.repeat_interleave(
        torch.arange(num_nodes, device=device),
        graph.forward_indptr.diff(),
    )
    forward_src = graph.forward_indices.long()
    forward_original = edge_index[:, graph.forward_edge_indices.long()]
    assert torch.equal(forward_src, forward_original[0])
    assert torch.equal(forward_dst, forward_original[1])

    backward_src = torch.repeat_interleave(
        torch.arange(num_nodes, device=device),
        graph.backward_indptr.diff(),
    )
    backward_dst = graph.backward_indices.long()
    backward_original = edge_index[:, graph.backward_edge_indices.long()]
    assert torch.equal(backward_src, backward_original[0])
    assert torch.equal(backward_dst, backward_original[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gatv2_edge_attr_uses_original_coo_edge_order() -> None:
    """Distinct edge IDs catch a missing or incorrect COO-to-CSR permutation."""
    device = torch.device("cuda")
    num_nodes, heads, head_dim = 5, 1, 32
    negative_slope = 0.13

    # Deliberately not sorted by destination (the forward CSR row key).
    edge_index = torch.tensor(
        [
            [3, 0, 4, 1, 2, 0, 3, 4],
            [1, 4, 0, 3, 1, 2, 4, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    graph = _build_cuda_graph(edge_index, num_nodes)

    x_left, x_right, attention, edge_attr = _make_inputs(
        num_nodes,
        edge_index.size(1),
        heads,
        head_dim,
        device=device,
        dtype=torch.float32,
        seed=11,
        requires_grad=False,
    )
    # Encode the original COO edge ID into one feature so that a wrong mapping
    # cannot accidentally pass because attributes are exchangeable.
    edge_attr.zero_()
    edge_attr[:, 0, 0] = torch.arange(edge_index.size(1), device=device) * 0.1 - 0.3

    actual = gatv2_aggr(
        graph,
        x_left,
        x_right,
        attention,
        edge_attr=edge_attr,
        negative_slope=negative_slope,
    )
    expected = gatv2_edge_attr_reference(
        edge_index,
        x_left,
        x_right,
        attention,
        edge_attr,
        num_nodes=num_nodes,
        negative_slope=negative_slope,
    )

    _assert_close(actual, expected, atol=1e-4, rtol=1e-4, name="COO edge-order forward")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("num_nodes", "heads", "head_dim", "quantile"),
    [
        (32, 1, 32, -1),   # all light nodes
        (64, 2, 32, 0.50), # light and heavy buckets
        (48, 4, 64, 0.75),
    ],
)
def test_gatv2_edge_attr_fp32_forward(
    num_nodes: int,
    heads: int,
    head_dim: int,
    quantile: float,
) -> None:
    device = torch.device("cuda")
    negative_slope = 0.17
    edge_index = _make_directed_graph(
        num_nodes,
        num_nodes * 5,
        device=device,
        seed=1000 + num_nodes + heads + head_dim,
    )
    graph = _build_cuda_graph(edge_index, num_nodes, quantile=quantile)
    x_left, x_right, attention, edge_attr = _make_inputs(
        num_nodes,
        edge_index.size(1),
        heads,
        head_dim,
        device=device,
        dtype=torch.float32,
        seed=20,
        requires_grad=False,
    )

    actual = gatv2_aggr(
        graph,
        x_left,
        x_right,
        attention,
        edge_attr=edge_attr,
        negative_slope=negative_slope,
    )
    expected = gatv2_edge_attr_reference(
        edge_index,
        x_left,
        x_right,
        attention,
        edge_attr,
        num_nodes=num_nodes,
        negative_slope=negative_slope,
    )

    assert torch.isfinite(actual).all(), "CUDA output contains NaN or Inf"
    _assert_close(actual, expected, atol=2e-4, rtol=2e-4, name="fp32 forward")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("num_nodes", "heads", "head_dim", "quantile"),
    [
        (32, 1, 32, -1),
        (64, 2, 32, 0.50),
        (48, 4, 64, 0.75),
    ],
)
def test_gatv2_edge_attr_fp32_backward_all_inputs(
    num_nodes: int,
    heads: int,
    head_dim: int,
    quantile: float,
) -> None:
    """Compare gradients for left/right node features, attention and edge_attr."""
    device = torch.device("cuda")
    negative_slope = 0.17
    edge_index = _make_directed_graph(
        num_nodes,
        num_nodes * 5,
        device=device,
        seed=2000 + num_nodes + heads + head_dim,
    )
    graph = _build_cuda_graph(edge_index, num_nodes, quantile=quantile)

    cuda_inputs = _make_inputs(
        num_nodes,
        edge_index.size(1),
        heads,
        head_dim,
        device=device,
        dtype=torch.float32,
        seed=30,
        requires_grad=True,
    )
    ref_inputs = tuple(value.detach().clone().requires_grad_(True) for value in cuda_inputs)

    cuda_out = gatv2_aggr(
        graph,
        *cuda_inputs[:3],
        edge_attr=cuda_inputs[3],
        negative_slope=negative_slope,
    )
    ref_out = gatv2_edge_attr_reference(
        edge_index,
        *ref_inputs,
        num_nodes=num_nodes,
        negative_slope=negative_slope,
    )

    generator = torch.Generator(device=device).manual_seed(31)
    grad_output = torch.randn(
        cuda_out.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    cuda_grads = torch.autograd.grad(cuda_out, cuda_inputs, grad_outputs=grad_output)
    ref_grads = torch.autograd.grad(ref_out, ref_inputs, grad_outputs=grad_output)

    names = ("grad_x_left", "grad_x_right", "grad_attention", "grad_edge_attr")
    for name, actual, expected in zip(names, cuda_grads, ref_grads, strict=True):
        assert actual is not None, f"{name} is None"
        assert torch.isfinite(actual).all(), f"{name} contains NaN or Inf"
        _assert_close(actual, expected, atol=4e-4, rtol=4e-4, name=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_zero_edge_attr_matches_no_edge_attr_for_node_and_attention_gradients() -> None:
    """The edge-aware branch with zeros must reduce to the old GATv2 path."""
    device = torch.device("cuda")
    num_nodes, heads, head_dim = 40, 2, 32
    negative_slope = 0.11
    edge_index = _make_directed_graph(num_nodes, num_nodes * 4, device=device, seed=40)
    graph = _build_cuda_graph(edge_index, num_nodes, quantile=0.5)

    with_edge_inputs = _make_inputs(
        num_nodes,
        edge_index.size(1),
        heads,
        head_dim,
        device=device,
        dtype=torch.float32,
        seed=41,
        requires_grad=True,
    )
    x_left, x_right, attention, edge_attr = with_edge_inputs
    edge_attr = torch.zeros_like(edge_attr, requires_grad=True)

    no_edge_inputs = tuple(value.detach().clone().requires_grad_(True) for value in (x_left, x_right, attention))

    out_with_edge = gatv2_aggr(
        graph,
        x_left,
        x_right,
        attention,
        edge_attr=edge_attr,
        negative_slope=negative_slope,
    )
    out_without_edge = gatv2_aggr(
        graph,
        *no_edge_inputs,
        edge_attr=None,
        negative_slope=negative_slope,
    )
    _assert_close(out_with_edge, out_without_edge, atol=1e-5, rtol=1e-5, name="zero-edge forward")

    generator = torch.Generator(device=device).manual_seed(42)
    grad_output = torch.randn(out_with_edge.shape, device=device, generator=generator)
    grads_with_edge = torch.autograd.grad(
        out_with_edge,
        (x_left, x_right, attention),
        grad_outputs=grad_output,
    )
    grads_without_edge = torch.autograd.grad(
        out_without_edge,
        no_edge_inputs,
        grad_outputs=grad_output,
    )

    for name, actual, expected in zip(
        ("grad_x_left", "grad_x_right", "grad_attention"),
        grads_with_edge,
        grads_without_edge,
        strict=True,
    ):
        _assert_close(actual, expected, atol=2e-4, rtol=2e-4, name=f"zero-edge {name}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gatv2_edge_attr_low_precision_forward_and_backward(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("The current CUDA device does not support bfloat16")

    device = torch.device("cuda")
    num_nodes, heads, head_dim = 32, 2, 32
    negative_slope = 0.2
    edge_index = _make_directed_graph(num_nodes, num_nodes * 4, device=device, seed=50)
    graph = _build_cuda_graph(edge_index, num_nodes, quantile=-1)

    low_precision_inputs = _make_inputs(
        num_nodes,
        edge_index.size(1),
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        seed=51,
        requires_grad=True,
    )
    fp32_inputs = tuple(value.detach().float().clone().requires_grad_(True) for value in low_precision_inputs)

    actual = gatv2_aggr(
        graph,
        *low_precision_inputs[:3],
        edge_attr=low_precision_inputs[3],
        negative_slope=negative_slope,
    )
    expected = gatv2_edge_attr_reference(
        edge_index,
        *fp32_inputs,
        num_nodes=num_nodes,
        negative_slope=negative_slope,
    )

    assert torch.isfinite(actual).all(), "Low-precision output contains NaN or Inf"
    _assert_close(actual, expected, atol=5e-2, rtol=5e-2, name=f"{dtype} forward")

    generator = torch.Generator(device=device).manual_seed(52)
    grad_output_fp32 = 0.2 * torch.randn(expected.shape, device=device, generator=generator)
    actual_grads = torch.autograd.grad(
        actual,
        low_precision_inputs,
        grad_outputs=grad_output_fp32.to(dtype),
    )
    expected_grads = torch.autograd.grad(
        expected,
        fp32_inputs,
        grad_outputs=grad_output_fp32,
    )

    for name, actual_grad, expected_grad in zip(
        ("grad_x_left", "grad_x_right", "grad_attention", "grad_edge_attr"),
        actual_grads,
        expected_grads,
        strict=True,
    ):
        assert torch.isfinite(actual_grad).all(), f"{dtype} {name} contains NaN or Inf"
        _assert_close(
            actual_grad,
            expected_grad,
            atol=2e-1,
            rtol=1e-1,
            name=f"{dtype} {name}",
        )
