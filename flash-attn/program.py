"""
This is the code that will test the implementation of our code.
"""

import torch
from zmq import device

from triton_attn import TritonAttention 

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    """
    Initialize the Q, K, V as normally distributed tensors
    causal param controls if attention should be causal or non-causal
    """
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)  # QK^T / sqrt(HEAD_DIM)
    dO = torch.randn_like(Q)  # Needed for backward pass

    ### REFERENCE IMPLEMENTATION ### 
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) 
    P = P * softmax_scale  # We did QK^T / sqrt(HEAD_DIM)

    if causal:
        P[:, :, MASK == 0] = float('-inf')

    # Apply softmax by rows
    # softmax will push -inf to 0 if causal
    P = torch.softmax(P.float(), dim = -1).half() 

    # reference outputs, note that this is naive implementation of attention
    ref_O = torch.matmul(P, V)
    ref_O.backward(gradient=dO)

    ref_dV, V.grad = V.grad.clone(), None  # type: ignore
    ref_dK, K.grad = K.grad.clone(), None  # type: ignore
    ref_dQ, Q.grad = Q.grad.clone(), None  # type: ignore

    ### TRITON IMPLEMENTATION ###
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half() # type: ignore
    tri_out.backward(gradient=dO)

    tri_dV, V.grad = V.grad.clone(), None  # type: ignore
    tri_dK, K.grad = K.grad.clone(), None  # type: ignore
    tri_dQ, Q.grad = Q.grad.clone(), None  # type: ignore

    # Compare (using only absolute difference)
    rtol, atol = 0.0, 1e-2
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)


if __name__ == '__main__':
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("Tests passed")



