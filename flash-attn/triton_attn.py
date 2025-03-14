import torch
import triton
import triton.language as tl


# A triton kernel is just a python method with the triton jit decorator
@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # Causal attn, STAGE passed 4 - (3) = 1
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        # i.e. the diagonal entries (remember all are blocks which carry individual elements)
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention, where we don't mask out anything
        lo, hi = 0, SEQ_LEN

    # Load blocks of keys and values
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):  # type: ignore
        # Let the compiler know (as a hint) that start_n is a multiple of BLOCK_N, so the compiler can do optimisations
        # NOTE: Telling the Triton compiler this information helps improve its pipelining algorithm for the 'for loop'
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # Compute qk
        # NOTE: K_block_ptr was already defined as a transpose so we take dot product directly
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(
            Q_block, K_block
        )  # Q_block was already loaded while calling _attn_fwd_inner

        # TODO: Add comments Timestamp 4:10:00
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])  # type: ignore
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale
            QK_block -= m_ij[:, None]

        # Compute the exponential of each dot product
        # so now we are computing exp(qk_ij - m_ij). We already subtracted m_ij in previous step
        P_block = tl.math.exp(QK_block)

        # Compute the sum by rows of the attention scores. We will 'fix' this normalisation factor (of the current block) later
        l_ij = tl.sum(P_block, 1)

        # Correction factor for the previous l_i. Same as e^(m_i (j-1) - m_i (j) )
        alpha = tl.ath.exp(m_i - m_ij)

        # Apply the correction factor to the previous l_i and add the new l_ij
        # l_i is the normalisation (correction) factor for each row in the current output block
        l_i = l_i * alpha + l_ij

        # Load the v block
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)  # type conversion

        # O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        # The O_block is used as an accummulator i.e. dot function needs some place to store intermediate results, so we use the O_block itself
        # The notation below is just optimised way
        # NOTE: "Matmul is just dot product which is in-turn repeated sum, the dot will keep summing the result
        O_block = tl.dot(P_block, V_block, acc=O_block)  # O_block += P_block @ V_block

        # m_i is the max value for each row, needed for the backward pass. We use this instead of again computing m_i in backward pass
        m_i = m_ij

        # Move to the next block of K and V by moving one BLOCK_SIZE_KV
        V_block_ptr = tl.advance(
            V_block_ptr, (BLOCK_SIZE_KV, 0)
        )  # NOTE Remember V is ptr to tensor of shape V[SEQ_LEN, HEAD_DIM]
        K_block_ptr = tl.advance(
            K_block_ptr, (0, BLOCK_SIZE_KV)
        )  # NOTE Remember K is already transposed K[HEAD_DIM, SEQ_LEN]

    return O_block, l_i, m_i


@triton.autotune(
    # NOTE: Triton on its own does not know which is the best block size for K, V etc
    # We need to calculate it depending on the hardware, for the best throughput
    # Num warps = num of blocks of 32 threads that work co-operatively running the same instruction at a time
    # Num stages = number of stages in software pipelining
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM  # noqa: E741
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

    # These pointers can also be treated directly as tensors, that is why we provide shapes

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

    # So far, we have moved our pointers to the right Q block and beginning of K, V block that we will work with

    # Now we need the offsets of queries inside of each block of queries that this program will work with
    # offsets_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offsets_k,v: the offsets for the tokens in the K and V sequence to process
    # Remember how initially, pointer to K and V point to beginning of sequence of K, V
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Now we will initialise variables
    # NOTE: Check line 5 of Algorithm

    # m_i: the running maximum. We have one for each query
    # This is a block of numbers based on how many queries we have in a block of queries
    # initialised with -inf
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    # The +1.0 is also in the original code. Maybe it is to make the log stable since later we use l_i to compute logsumexp
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # acc: the accumulator for the output, which is a group of rows of the 0 matrix
    # NOTE: This is one a row (block) of the O matrix. Check "readme-images/bmm4.png
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Load the blocks of Q: it will stay in SRAM throughout, we took from HBM
    Q_block = tl.load(Q_block_ptr)  # block_shape=(BLOCK_SIZE_Q, HEAD_DIM),

    # Stage: 3 if causal else 1

    # The _attn_fwd_inner will be a for loop
    # NOTE: This inner loop will go over over all key and value blocks one by one
    # And for each kv block, it will have to fix previously computed SOFTMAX* block
    # Fix will be by the diag matrix in algorithm from previous iteration.
    # Base output is P_i * V_i from the current iteration

    # NOTE: Check the splitting of for loop in readme
    # Causal case:
    # Loop 1 --> STAGE = 3 --> _attn_fwd_inner with 4 - STAGE = 1 --> From 0 to the left of the diagonal
    # Loop 2 --> Next loop with 2 --> diagonal entries
    # Non-causal case:
    # Loop 1 --> STAGE = 1 --> _attn_fwd_inner with 4 - STAGE = 3 --> All elements i.e. else condition

    # First, we run the loop for everything before diagonal (needed for both causal and non causal)
    # Then if causal, we skip run the loop for everything after diagonal
    if STAGE == 1 or STAGE == 3:
        # Run for all where key index is smaller than current queries block
        # This step runs for the blocks to the left of the diagonal
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4
            - STAGE,  # tells if we are on left or right of diagonal - do we apply causal mask or not
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # When key index is more than query index, we don't need to compute anything since it will be 0 right of diag
        # This step runs for the blocks in the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    # This is needed to compute the logsumexp for the backward pass
    # For backward pass to recompute softmax without having to recalculate normalisation factor and max value for
    # each row, we should be saving both i.e.
    # max value of each row in query block and
    # normalisation factor for each query in the query block
    m_i += tl.math.log(l_i)

    # Normalise the output at the end
    O_block = O_block / l_i[:, None]

    # Save the normalisation factor and max value for the backward pass
    # Remember M is ptr to tensor of shape BATCH_SIZE, NUM_HEADS, SEQ_LEN
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_type))


@triton.jit
def _attn_bwd_preprocess(
    O,  # ptr to O
    dO,  # ptr to dO
    D,  # ptr to D
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # NOTE: Remember D: (BATCH_SIZE, NUM_HEADS, SEQ_LEN) i.e. one for each member of output -- same shape as M
    # NOTE: Q and O have same shape. Infact shape of O depends on shape of Q
    # NOTE: Number of Query Vectors = SEQ_LEN

    # program id at axis 0 --> from launch grid --> SEQ_LEN // BLOCK_SIZE_MACRO --> of O
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(
        0, BLOCK_SIZE_Q
    )  # consider skipping terms

    # program id at axis 1 --> from launch grid --> BATCH_SIZE * NUM_HEADS
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    # NOTE: This time we won't use make_block_ptr, instead use stride

    # Load a single block of BLOCK_SIZE_Q rows of O
    # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    # Batch 0 and Head 0 will have SEQ_LEN * HEAD_DIM items
    # Batch 0 and Head 1 also same ...
    # How many items to skip to reach for next head's elements = index_batch_head * HEAD_DIM * SEQ_LEN
    # then select a 2-dimensional tensor where offsets of rows are offs_q and for cols it is offs_dim
    O_block = tl.load(
        pointer=O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    )  # Shape (BLOCK_SIZE_Q, HEAD_DIM)

    # Remember how O and Q have same shape
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)

    # NOTE: From paper algo Line 19: compute D_i
    # NOTE: D matrix contains all rows D_i
    # Element-wise product of dO_i * O_i
    D_block = tl.sum(dO_block * O_block, axis=1)  # (BLOCK_SIZE_Q, )
    # Store the D block. Each batch will have NUM_HEADS * SEQ_LEN elements
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block_ptrs)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    # offsets by which we need to move the Q, K, V to right batch and right head in the batch
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = (
        index_batch_head % NUM_HEADS
    )  # this is similar as done in the forward pass
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )  # right batch and right head

    # This is the offset that allows us to select the right sequence given the batch and head.
    # Only applicable for M and D - only they don't have the head dim, only batch_size, num_heads, seq_len,
    # Consider SEQ_LEN as the stride to move from one batch head to next batch head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    # All have shape [B, NUM_HEADS, SEQ_LEN, HEAD_DIM] -> skipping to right (first)batch and head
    # i.e. [0, 0, SEQ_LEN, HEAD_DIM]
    # [0, 0, start_kv: start_kv + BLOCK_KV, 0: HEAD_DIM]
    # [0, 0, start_kv: start_kv + BLOCK_KV, offs_dim]
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head  # all dO, dQ, dK, dV have the same shape
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    # Now pointing to the first vector of seq of first batch and first head
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    # This might skik some KVs that will already be managed by other programs maybe in parallel
    # NOTE: See how grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
    # We were fixing the BLOCK_SIZE_MACRO
    # So program_id 0 is the number of BLOCK_SIZE_MACRO KVs that are already being managed by others
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV  # our KVs start from start_kv

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)  # BLOCK_KV is equal to BLOCK_SIZE_MACRO

    # Initialise with zeros
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop
    # NOTE: K is already pointing to the right tensor as we added the offsets above
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)  # 0, 1, 2, ... 32
    # After skipping becomes 0, 128, 2*128, 3*128, ...
    # For simplicity lets assume 4 dimensinos instead of 128
    # 0, 4, 2*4, 3*4
    # 0   + (0,1,2,3) = (0,1,2,3)
    # 4   + (0,1,2,3) = (4,5,6,7)
    # 2*4 + (0,1,2,3) = (8,9,10,11)
    # 3*4 + (0,1,2,3) = (12,13,14,15)

    # Taken AS IS
    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector ans offs_dim as a row vector
    # This is equivalent to doing offs_q.unsqueeze(1) i.e. adding the column dimension
    ## q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    ## qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    # NOTE: We need only Q in transpose and not O (given in the paper above appendix 5)
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Iterates over the sequence dimension of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        # Objective is to compute P_T values on the fly
        offs_q = curr_q + tl.arange(0, BLOCK_Q)

        # Remember M we got from the forward pass
        m = tl.load(M + offs_q)

        # P^T = softmax(S^T)
        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = S^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        # Mask out values in causal case
        if STAGE == 3:
            # Autoregressive masking.
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )  # Shape: (BLOCK_KV1, BLOCK_Q1)
            # Replace all the masked values with 0.
            # In this case we do not need to mask with -Inf before applying the softmax since we already computed the normalization factors (stored in "m")
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Delta = rowsum(O * dO) where * is the element-wise product
        Di = tl.load(D + offs_q)

        # NOTE: dK is the required output for the next few steps
        # Given in line 22 of paper's algo
        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Write back dV.
    # dV is already pointing to the right batch and right head because we incremented it earlier
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK.
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    # TODO: Copied as-is
    # First part is copying the Q, K, V pointers to the right place
    # Same as done earlier in the previous for loop
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    # exact starting point where this program should be working with
    # We have two dimensions - batch and head; which query among the full sequene length
    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Treat offs_q as a column vector and repeat it along the row vector
    # Where each column will be head dimension * stride
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    # We access the K and V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    # Till here it was very similar to the previous function

    for blk_idx in range(num_steps):
        # Load as usual in SRAM
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)  # subtract the logsumexp value

        if STAGE == 3:
            # Causal attention so Autoregressive masking.
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS.
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        # Compute dQ. As specified in the paper
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Increment pointers.
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


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
        )  # type: ignore

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = (
            ctx.saved_tensors
        )  # all tensors that we saved during forward pass
        # to optimise memory usage, we don't save QK^T matrix directly - would be too big
        # They are pointing to the beginning of
        # first batch, first head, first token, first dimension

        # NOTE:Pytorch during Autograd will give us d(loss)/d(output)

        assert dO.is_contihuous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        # NOTE: Remember how shape of derivative is same as shape of vector wrt whom
        # gradient was taken
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        # NOTE: NUM_WARPS: Num of threads, NUM_STAGES: Stages in Software Pipelining
        # NOTE: Macro is what we fix and Micro is what we iterate upon
        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        # Preprocess kernel
        # preprocess_grid is the launch grid of the kernel _attn_bwd_preprocess
        # Launched independently for each batch and each head
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
        # D_i values are one for each vector in O tensor

        # Compute all the elements
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        # launch grid for next iteration
        # So grid is defined for BLOCK_SIZE_MACRO (i.e. fixed thing)
        # Within that grid, we use BLOCK_SIZE_MICRO
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1  # Same 3 for causal 1 for non-causal

        # Fix KV and iterate through all the Q blocks in MICRO size (32)
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,  # type: ignore
            num_stages=NUM_STAGES,  # type: ignore
        )

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,  # type: ignore
            num_stages=NUM_STAGES,  # type: ignore
        )

        return dQ, dK, dV, None, None
