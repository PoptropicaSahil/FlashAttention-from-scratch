import torch

import triton


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        """
        ctx is the context. We save forward pass info (like activations,
        normalisation factor etc) as context to use during the backward pass
        """
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # NOTE: Shape of output depends on the shape of Q sequence
        # because when cross attention, seq len of K and Q differs
        O = torch.empty_like(Q)

        stage = 3 if causal else 1

        # Remember outer loop is over Q blocks - all independent - so we can spawn as many programs as blocks of Q
        # each block of queries is a group of tokens in the query sequence
        # divide the query into blocks of queries
        # We are launching our program along two dimensions
        # Grid tells us how many grids in parallel to launch - if all available then nice else GPU decides how many sequentially
        grid = lambda args: (  # noqa: E731, F841
            triton.cdiv(
                x=SEQ_LEN, y=args["BLOCK_SIZE_Q"]
            ),  # Which group of queries (=seq_len/block_size) do we work with
            BATCH_SIZE
            * NUM_HEADS,  # Which head of which batch element are we going to work with
            1,  # Z dimension in the CUDA launch grid
        )

        # M is the logsumexp for the backward pass, one for each query
        # Usually in forward pass we'll store two things - row max and normalisation factor
        # But with logsumexp trick, we can only store one value i.e. L_i
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
