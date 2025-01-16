import torch

import triton
import triton.language as tl


# A triton kernel is just a python method with the triton jit decorator
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,  # these were not passed when calling the method because they will be passed when we apply the auto tuning decorator
    BLOCK_SIZE_KV: tl.constexpr,  # similarly
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # One program -- which is identified by program_id(0) on first axis and program_id(1) on second axis
    # Each of such programs will run independently in parallel
    # Each of them will have a different value for block_index_q and index_batch_head

    # This indicates which block in the sequence length to process
    # This one block of queries will produce one block of output
    block_index_q = tl.program_id(0)

    # Which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # Which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # Position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # Now enter the program at the right location. Note that the pointers Q, K, V are pointing to the beginning of the tensors
    # Allows to get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by selecting indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # Till now we are pointing to the correct batch and the head i.e. Q[index_batch, index_head, :, :]
    # Q Shape is [index_batch, index_head, seq_len, head_dim]
    # We are pointing to the first element of tensor made of seq_len and head_dim
    # Inside of this tensor, we have to choose the right block of query the program should work with
    # But seq_len is all the queries
    # So we need to skip some queries to reach exact next query block --> Given by offsets = block_index_q * BLOCK_SIZE_Q
    # After skipping we are at 
    # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,  # The base pointer to the parent tensor
        shape=(SEQ_LEN, HEAD_DIM),  # Create block of shape of last two dimensions
        strides=(stride_Q_seq, stride_Q_dim),  # Note how strides align with the shape
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),  # Don't know what this does
    )

    # No offsets here because a new program will process contents of outer for loop of Q. Within each loop,
    # we use (all) K, V inside
    # NOT ALL, but only the number of elements indicated by the 'block_shape' parameter of each pointers block.
    # Consider each pointers block to be a tensor of pointers with the shape as 'block_shape'
    # Pointing at V[index_batch, index_head, :, :]
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # K[index_batch, index_head, :, :]
    # Again, rememeber we are not selecting all K, but a block of K
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),  # note the transposed shape
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # invert the strides wrt Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    # O[index_batch, index_head, :, :]
    # Since output has same shape as queries, skipping (offsets) is in terms of query
    # Becomes O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )


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
        # Grid tells us how many programs in parallel to launch - if all available then nice else GPU decides how many sequentially
        grid = (  # noqa: E731
            lambda args: (  # noqa: F841
                triton.cdiv(
                    x=SEQ_LEN, y=args["BLOCK_SIZE_Q"]
                ),  # Which group of queries (=ceil of seq_len/block_size_qss) do we work with
                BATCH_SIZE
                * NUM_HEADS,  # Which head of which batch element are we going to work with
                1,  # Z dimension in the CUDA launch grid
            )
        )

        # M is the logsumexp for the backward pass, one for each query
        # Usually in forward pass we'll store two things - row max and normalisation factor
        # But with logsumexp trick, we can only store one value i.e. L_i
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # launch kernel for fwd pass
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,  # actually L in the code of flashattention
            O=O,
            # We are passing the stride because we only get pointers in triton, usual indexing isn't possible
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,  # causal or non causal attention
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return O
