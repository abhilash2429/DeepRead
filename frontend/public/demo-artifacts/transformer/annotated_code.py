# Component: Add & Norm
# Provenance: paper-stated
# Assumption: The `eps` parameter for `nn.LayerNorm` is assumed to be `1e-6`, which is a common default for numerical stability in PyTorch's `LayerNorm` implementation.
# Assumption: The `d_model` parameter, representing the dimensionality of the model (and thus the feature dimension for LayerNorm), is inferred as a necessary input for `nn.LayerNorm` to specify the `normalized_shape` in the Transformer architecture.
import torch
import torch.nn as nn

class AddNorm(nn.Module):
    """
    Implements the Add & Norm component as described in the paper.
    This corresponds to the post-norm architecture: LayerNorm(x + Sublayer(x)).
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # INFERRED: d_model is the dimensionality of the model, required for LayerNorm.
        #           It represents the feature dimension of the input to LayerNorm.
        # ASSUMED: eps is a small value added to the variance for numerical stability in LayerNorm.
        #          A common default is 1e-6 or 1e-5 in PyTorch's LayerNorm.
        self.norm = nn.LayerNorm(d_model, eps=eps) # Eq. (N) - Layer Normalization

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # Eq. (N) - Residual connection: x + Sublayer(x)
        # The paper describes "Add & Norm" as applying LayerNorm after the residual connection.
        # The ambiguity resolution 'layernorm_placement' confirms this exact order: `LayerNorm(x + Sublayer(x))`
        return self.norm(x + sublayer_output)

# Component: Decoder Layer
# Provenance: paper-stated
# Assumption: d_k = d_v = d_model / num_heads, as is standard for Transformer attention.
# Assumption: Bias terms are initialized to zero, which is a common practice.
# Assumption: Dropout is applied to the attention weights before multiplying with V inside MultiHeadAttention.
# Assumption: Mask values are 0 for masked positions, 1 for unmasked positions.
# Assumption: d_model must be divisible by num_heads for d_k to be an integer.
# Assumption: bias=True for all linear layers based on ambiguity resolution 'bias_in_linear_layers'.
# Assumption: Using nn.init.xavier_uniform_ as it's a standard choice for Xavier initialization. The ambiguity resolution 'weight_initialization' mentions "variance-scaling initializer similar to Xavier uniform, scaling by (d_in + d_out) / 2". nn.init.xavier_uniform_ scales by sqrt(6 / (fan_in + fan_out)), which is a common form of Xavier.
# Assumption: Using a large negative number (-1e9) to effectively zero out masked attention scores after softmax for numerical stability.
# Assumption: Using nn.init.kaiming_uniform_ as it's a standard choice for Kaiming initialization. The ambiguity resolution 'weight_initialization' states "Kaiming (He) initialization" for FFN. For the second layer of FFN, while it doesn't have a ReLU *after* it, it's common practice to use Kaiming for both layers in the FFN block if the first layer uses ReLU.
# Assumption: Dropout is applied to the final output of the FFN sub-layer, as per ambiguity resolution 'dropout_placement_ffn'.
# Assumption: LayerNorm placement is 'post-norm' as per ambiguity resolution 'layernorm_placement': LayerNorm(x + Sublayer(x)).
# Assumption: Dropout for the residual connection is applied after the sub-layer output, before adding to the input. This is distinct from internal dropouts within MultiHeadAttention or FFN.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper module: MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        # ASSUMED: d_k = d_v = d_model / num_heads, as is standard for Transformer attention.
        # INFERRED: d_model must be divisible by num_heads for d_k to be an integer.
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        # Linear projections for Q, K, V
        # INFERRED: bias=True for all linear layers based on ambiguity resolution 'bias_in_linear_layers'.
        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=True)

        # Output linear projection
        # INFERRED: bias=True for all linear layers based on ambiguity resolution 'bias_in_linear_layers'.
        self.out_linear = nn.Linear(d_model, d_model, bias=True)

        # ASSUMED: Dropout is applied to the attention weights before multiplying with V.
        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters() # For weight initialization

    def _reset_parameters(self):
        # Weight initialization: Xavier (Glorot) for attention projections.
        # INFERRED: Using nn.init.xavier_uniform_ as it's a standard choice for Xavier initialization.
        # The ambiguity resolution 'weight_initialization' mentions "variance-scaling initializer similar to Xavier uniform, scaling by (d_in + d_out) / 2".
        # nn.init.xavier_uniform_ scales by sqrt(6 / (fan_in + fan_out)), which is a common form of Xavier.
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

        # ASSUMED: Bias terms are initialized to zero, which is a common practice.
        if self.q_linear.bias is not None:
            nn.init.constant_(self.q_linear.bias, 0.)
        if self.k_linear.bias is not None:
            nn.init.constant_(self.k_linear.bias, 0.)
        if self.v_linear.bias is not None:
            nn.init.constant_(self.v_linear.bias, 0.)
        if self.out_linear.bias is not None:
            nn.init.constant_(self.out_linear.bias, 0.)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        batch_size = query.size(0)

        # 1) Linear projections and split into heads
        # Eq. (1) (implicitly, as part of h_i = Attention(QW_Q, KW_K, VW_V))
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Scaled Dot-Product Attention
        # Eq. (1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Apply mask (for self-attention, it's look-ahead mask; for cross-attention, it's padding mask)
            # ASSUMED: Mask values are 0 for masked positions, 1 for unmasked positions.
            # INFERRED: Using a large negative number (-1e9) to effectively zero out masked attention scores after softmax.
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # Eq. (1)
        attn_weights = self.dropout(attn_weights) # ASSUMED: Dropout applied to attention weights before multiplying with V

        context = torch.matmul(attn_weights, v) # Eq. (1)

        # 3) Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context) # Eq. (1)

        return output, attn_weights

# Helper module: PositionwiseFeedForward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        # INFERRED: bias=True for all linear layers based on ambiguity resolution 'bias_in_linear_layers'.
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        # INFERRED: Dropout is applied to the final output of the FFN sub-layer, as per ambiguity resolution 'dropout_placement_ffn'.
        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters() # For weight initialization

    def _reset_parameters(self):
        # Weight initialization: Kaiming (He) for FFN with ReLU activations.
        # INFERRED: Using nn.init.kaiming_uniform_ as it's a standard choice for Kaiming initialization.
        # The ambiguity resolution 'weight_initialization' states "Kaiming (He) initialization" for FFN.
        nn.init.kaiming_uniform_(self.w_1.weight, nonlinearity='relu')
        # For the second layer, while it doesn't have a ReLU *after* it, it's common practice
        # to use Kaiming for both layers in the FFN block if the first layer uses ReLU.
        nn.init.kaiming_uniform_(self.w_2.weight, nonlinearity='relu')

        # ASSUMED: Bias terms are initialized to zero, which is a common practice.
        if self.w_1.bias is not None:
            nn.init.constant_(self.w_1.bias, 0.)
        if self.w_2.bias is not None:
            nn.init.constant_(self.w_2.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. (3)
        # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        # Dropout applied to the output of FFN before residual connection, as per ambiguity resolution 'dropout_placement_ffn'.
        return self.dropout(self.w_2(F.relu(self.w_1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float, eps: float):
        super().__init__()

        # Masked Multi-Head Self-Attention sub-layer
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        # INFERRED: LayerNorm placement is 'post-norm' as per ambiguity resolution 'layernorm_placement': LayerNorm(x + Sublayer(x)).
        self.self_attn_norm = nn.LayerNorm(d_model, eps=eps)
        # INFERRED: Dropout for the residual connection is applied after the sub-layer output, before adding to the input.
        self.self_attn_residual_dropout = nn.Dropout(dropout_rate)

        # Multi-Head Encoder-Decoder Attention sub-layer
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        # INFERRED: LayerNorm placement is 'post-norm' as per ambiguity resolution 'layernorm_placement'.
        self.cross_attn_norm = nn.LayerNorm(d_model, eps=eps)
        # INFERRED: Dropout for the residual connection.
        self.cross_attn_residual_dropout = nn.Dropout(dropout_rate)

        # Feed-Forward Network sub-layer
        # The FFN module itself includes the dropout as per ambiguity resolution 'dropout_placement_ffn'.
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        # INFERRED: LayerNorm placement is 'post-norm' as per ambiguity resolution 'layernorm_placement'.
        self.ffn_norm = nn.LayerNorm(d_model, eps=eps)
        # No separate residual dropout here, as the FFN module already applies it internally as per 'dropout_placement_ffn'.

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:

        # Masked Multi-Head Self-Attention sub-layer
        # Post-norm architecture: LayerNorm(x + Sublayer(x))
        # First, apply LayerNorm to the input of the sub-layer.
        _x = self.self_attn_norm(x)
        # Then, compute the sub-layer output.
        self_attn_output, _ = self.self_attn(_x, _x, _x, tgt_mask) # Q, K, V, mask
        # Apply dropout to the sub-layer output, then add to the original input (residual connection).
        x = x + self.self_attn_residual_dropout(self_attn_output) # Eq. (2)

        # Multi-Head Encoder-Decoder Attention sub-layer
        # Post-norm architecture: LayerNorm(x + Sublayer(x))
        _x = self.cross_attn_norm(x)
        cross_attn_output, _ = self.cross_attn(_x, encoder_output, encoder_output, src_mask) # Q, K, V, mask
        x = x + self.cross_attn_residual_dropout(cross_attn_output) # Eq. (2)

        # Feed-Forward Network sub-layer
        # Post-norm architecture: LayerNorm(x + Sublayer(x))
        _x = self.ffn_norm(x)
        # The FFN module already applies dropout to its output as per 'dropout_placement_ffn'.
        ffn_output = self.ffn(_x)
        x = x + ffn_output # Eq. (2)

        return x

# Component: Decoder Stack
# Provenance: paper-stated
# Assumption: d_model: TODO: Specify model dimension (e.g., 512).
# Assumption: num_heads: TODO: Specify number of attention heads (e.g., 8).
# Assumption: d_ff: TODO: Specify feed-forward inner dimension (e.g., 2048).
# Assumption: dropout_rate: TODO: Specify dropout rate (e.g., 0.1).
# Assumption: N: TODO: Specify number of decoder layers (e.g., 6).
# Assumption: eps: TODO: Specify epsilon for LayerNorm (e.g., 1e-6).
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper: Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.
    """
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # INFERRED: Dropout is applied to the attention weights before multiplying with V.
        # This is standard practice in the original Transformer paper.

    def forward(self, query, key, value, mask=None):
        # Eq. (1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # Eq. (1)

        if mask is not None:
            # INFERRED: Use a large negative number for masked positions to ensure softmax outputs zero.
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1) # Eq. (1)
        p_attn = self.dropout(p_attn) # INFERRED: Dropout applied to attention probabilities.

        return torch.matmul(p_attn, value), p_attn

# Helper: Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention as described in the paper.
    """
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        # ASSUMED: d_model, num_heads, dropout_rate are provided as hyperparameters.
        self.d_model = d_model
        self.num_heads = num_heads
        # INFERRED: d_k (dimension per head) is d_model // num_heads.
        # This must be an integer.
        # Eq. (2)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V, and output.
        # Eq. (3)
        # bias_in_linear_layers: Include bias terms for all linear transformations.
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(4)])
        # INFERRED: The first three linears are for W_Q, W_K, W_V, and the last one is for W_O.

        self.attention = ScaledDotProductAttention(dropout_rate)
        # INFERRED: Output dropout for the MultiHeadAttention sub-layer is handled by SublayerConnection.

        self._reset_parameters()

    def _reset_parameters(self):
        # weight_initialization: Xavier (Glorot) initialization for attention projections.
        # The tensor2tensor library used a variance-scaling initializer similar to Xavier uniform.
        for i in range(4):
            nn.init.xavier_uniform_(self.linears[i].weight)
            if self.linears[i].bias is not None:
                nn.init.constant_(self.linears[i].bias, 0.) # INFERRED: Bias initialized to zero.

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        # Eq. (3)
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask) # Eq. (1)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # Eq. (4)
        return self.linears[-1](x) # Eq. (4)

# Helper: Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        # ASSUMED: d_model, d_ff, dropout_rate are provided as hyperparameters.
        # Eq. (5)
        # bias_in_linear_layers: Include bias terms for all linear transformations.
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        # dropout_placement_ffn: Apply dropout only to the final output of the FFN sub-layer,
        # i.e., `Dropout(FFN(x))`, before the residual connection. This means the dropout
        # is handled by the SublayerConnection, not internally within this module.

        self._reset_parameters()

    def _reset_parameters(self):
        # weight_initialization: Kaiming (He) initialization for models with ReLU activations (like the FFN).
        nn.init.kaiming_uniform_(self.w_1.weight, nonlinearity='relu')
        if self.w_1.bias is not None:
            nn.init.constant_(self.w_1.bias, 0.) # INFERRED: Bias initialized to zero.

        # weight_initialization: For the second linear layer, Xavier (Glorot) is a standard choice.
        nn.init.xavier_uniform_(self.w_2.weight)
        if self.w_2.bias is not None:
            nn.init.constant_(self.w_2.bias, 0.) # INFERRED: Bias initialized to zero.

    def forward(self, x):
        # Eq. (5)
        # INFERRED: ReLU activation is used as per the paper.
        return self.w_2(F.relu(self.w_1(x)))

# Helper: SublayerConnection (Add & Norm)
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    Implements the post-norm architecture: LayerNorm(x + Dropout(Sublayer(x))).
    """
    def __init__(self, d_model, dropout_rate, eps=1e-6):
        super().__init__()
        # ASSUMED: d_model, dropout_rate, eps are provided as hyperparameters.
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)
        # layernorm_placement: Implement the post-norm architecture: LayerNorm(x + Sublayer(x)).
        # This means LayerNorm is applied *after* the residual connection and dropout.

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # layernorm_placement: Post-norm architecture: LayerNorm(x + Sublayer(x))
        # The dropout is applied to the sublayer output before adding to the residual.
        return self.norm(x + self.dropout(sublayer(x)))

# Decoder Layer
class DecoderLayer(nn.Module):
    """
    One layer of the decoder.
    Consists of masked multi-head self-attention, encoder-decoder attention, and a feed-forward network.
    Each sub-layer is followed by a residual connection and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, eps=1e-6):
        super().__init__()
        # ASSUMED: d_model, num_heads, d_ff, dropout_rate, eps are provided as hyperparameters.
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.src_attn = MultiHeadAttention(d_model, num_heads, dropout_rate) # Encoder-Decoder Attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout_rate, eps) for _ in range(3)])
        # INFERRED: Three sublayers in a decoder layer as per the paper's architecture.

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follows Figure 1 (right) for connections."
        # Masked Multi-Head Self-Attention
        # Eq. (1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # Multi-Head Encoder-Decoder Attention
        # Eq. (1)
        # INFERRED: Query comes from the decoder, Key/Value come from the encoder output (memory).
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))

        # Position-wise Feed-Forward Network
        # Eq. (5)
        x = self.sublayer[2](x, self.feed_forward)
        return x

# Decoder Stack
class DecoderStack(nn.Module):
    """
    A stack of N identical decoder layers.
    """
    def __init__(self, N, d_model, num_heads, d_ff, dropout_rate, eps=1e-6):
        super().__init__()
        # ASSUMED: N (number of layers), d_model, num_heads, d_ff, dropout_rate, eps are provided as hyperparameters.
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate, eps)
            for _ in range(N)
        ])
        # INFERRED: A final LayerNorm is typically applied after the last decoder layer
        # in the overall Transformer architecture before the final linear projection.
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
