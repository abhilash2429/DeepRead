# Component: BERT Model (Multi-layer Transformer Encoder)
# Provenance: paper-stated
# Assumption: A01: Position embeddings are implemented as a learned lookup table of size (max_sequence_length, hidden_size), as specified in the resolved ambiguity. They are trained from scratch.
# Assumption: The specific values for hyperparameters like `vocab_size`, `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `intermediate_size`, `max_position_embeddings`, `type_vocab_size`, `hidden_dropout_prob`, `attention_probs_dropout_prob`, and `layer_norm_eps` are not provided in the prompt. The code is designed to accept these as arguments. For a typical BERT-base model, these would be, for example: `vocab_size=30522`, `hidden_size=768`, `num_attention_heads=12`, `num_hidden_layers=12`, `intermediate_size=3072`, `max_position_embeddings=512`, `type_vocab_size=2`, `hidden_dropout_prob=0.1`, `attention_probs_dropout_prob=0.1`, `layer_norm_eps=1e-12`.
import torch
import torch.nn as nn
import math

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # ASSUMED: A01 - Position embeddings are a learned lookup table.
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # INFERRED: Position IDs are usually generated internally if not provided.
        # This is a common practice in HuggingFace Transformers based on the original BERT implementation.
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if input_ids is None:
            # TODO: Handle case where input_ids is None, perhaps raise an error or return zero embeddings.
            # For now, assume input_ids is always provided.
            raise ValueError("input_ids must be provided for BertEmbeddings.")

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            # INFERRED: If position_ids are not provided, generate them based on sequence length.
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            # INFERRED: If token_type_ids are not provided, assume all zeros (segment A).
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_size)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Eq. (1)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # Eq. (1) scaling
        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # Eq. (2)

        # This is actually dropping out entire tokens to attend to, which might
        # make more sense for attention modules than dropping individual attention
        # scores.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # Eq. (3)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # Residual connection + LayerNorm
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # INFERRED: BERT uses GELU activation function, as per the original implementation.
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # Residual connection + LayerNorm
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps, intermediate_size):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps, intermediate_size):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(
                hidden_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                layer_norm_eps,
                intermediate_size
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class BertModel(nn.Module):
    """
    The bare BERT Model transformer outputting raw hidden-states without any specific head on top.
    """
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
                 intermediate_size, max_position_embeddings, type_vocab_size,
                 hidden_dropout_prob, attention_probs_dropout_prob, layer_norm_eps):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
            hidden_dropout_prob, layer_norm_eps
        )
        self.encoder = BertEncoder(
            num_hidden_layers, hidden_size, num_attention_heads,
            attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps,
            intermediate_size
        )

        # INFERRED: Pooler layer for sequence classification tasks, often used for [CLS] token output.
        # This is part of the standard BERT architecture, though not strictly "encoder" output.
        # It's a linear layer followed by tanh activation, as per the original BERT implementation.
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        # INFERRED: Standard BERT weight initialization, typically a truncated normal distribution.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        if input_ids is None:
            # TODO: Handle case where input_ids is None, perhaps raise an error.
            raise ValueError("input_ids must be provided for BertModel.")

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # This attention mask is more simple than the triangular one used in causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast here.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Extended attention mask for broadcasting
        # (batch_size, 1, 1, seq_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw attention scores in the self-attention layer,
        # the softmax will be nearly zero for the masked positions.
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )

        # Pooler output for [CLS] token
        # INFERRED: The pooler takes the hidden state of the first token ([CLS])
        # and applies a linear layer followed by a Tanh activation.
        first_token_tensor = encoder_output[:, 0]
        pooled_output = self.pooler(first_token_tensor)
        pooled_output = self.pooler_activation(pooled_output)

        return encoder_output, pooled_output

# Component: Classification Head (Fine-tuning)
# Provenance: inferred
# Assumption: The architecture of the classification head (dropout layer followed by a linear layer) is inferred based on common practices for fine-tuning BERT-like models for sequence classification, as no specific architecture was detailed in the provided context.
# Assumption: A default `dropout_prob` of 0.1 is assumed if not explicitly provided, consistent with typical BERT fine-tuning configurations.
# Assumption: The `hidden_size` parameter, representing the BERT model's hidden state dimension, is a placeholder and must be provided from the specific BERT model configuration (e.g., 768 for BERT-base).
# Assumption: The `num_labels` parameter, representing the number of classes for the downstream task, is a placeholder and must be provided based on the specific task requirements (e.g., 2 for binary classification).
import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Classification Head for fine-tuning BERT on sequence classification tasks.
    It takes the pooled output (usually the [CLS] token's representation)
    and projects it to the number of output labels.
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = None):
        super().__init__()
        # INFERRED: A dropout layer is typically applied before the final classification layer
        # for regularization during fine-tuning, as seen in official BERT implementations.
        # ASSUMED: If dropout_prob is not provided, a common default of 0.1 is used,
        # consistent with BERT's pre-training and fine-tuning practices.
        self.dropout = nn.Dropout(dropout_prob if dropout_prob is not None else 0.1)
        
        # INFERRED: A linear layer is used to project the hidden state of the [CLS] token
        # to the number of output classes for the specific classification task.
        self.classifier = nn.Linear(hidden_size, num_labels)

        # TODO: hidden_size - The dimension of the BERT model's hidden states.
        # This value is typically 768 for BERT-base and 1024 for BERT-large.
        # It must be provided from the BERT model configuration. # ASSUMED
        
        # TODO: num_labels - The number of classes for the specific downstream classification task.
        # This value is task-dependent (e.g., 2 for binary classification, N for multi-class). # ASSUMED

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classification head.

        Args:
            pooled_output (torch.Tensor): The pooled output from the BERT model,
                                          typically the final hidden state of the [CLS] token.
                                          Shape: (batch_size, hidden_size)

        Returns:
            torch.Tensor: Logits for each class. Shape: (batch_size, num_labels)
        """
        # Apply dropout for regularization
        x = self.dropout(pooled_output)
        
        # Project the hidden state to the number of output labels
        logits = self.classifier(x)
        
        return logits

# Component: Final Hidden States (T_i)
# Provenance: inferred
# Assumption: Standard BERT architecture components are used as described in the paper.
# Assumption: A01: The position embeddings are a learned lookup table of size (max_sequence_length, hidden_size), e.g., (512, 768). They are trained from scratch along with the rest of the model. For sequences longer than 512, a common strategy is to truncate the input, though this is not specified in the paper.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as Math # For Math.sqrt

# ASSUMED: Standard BERT architecture components are used as described in the paper.
# INFERRED: The final hidden states (T_i) are the output of the last layer of the Transformer encoder.

def gelu(x):
    """
    Original Google BERT implementation uses this approximation of GELU.
    """
    return x * 0.5 * (1.0 + torch.erf(x / Math.sqrt(2.0))) # Eq. (GELU approximation)

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self):
        super().__init__()
        self.vocab_size = TODO_VOCAB_SIZE # INFERRED: From BERT pre-training, e.g., 30522 for uncased base.
        self.hidden_size = TODO_HIDDEN_SIZE # INFERRED: From BERT base config, e.g., 768.
        self.max_position_embeddings = TODO_MAX_POSITION_EMBEDDINGS # ASSUMED: A01, e.g., 512.
        self.type_vocab_size = TODO_TYPE_VOCAB_SIZE # INFERRED: From BERT config, usually 2 (segment A, segment B).
        self.hidden_dropout_prob = TODO_HIDDEN_DROPOUT_PROB # INFERRED: From BERT config, e.g., 0.1.
        self.layer_norm_eps = TODO_LAYER_NORM_EPS # INFERRED: From BERT config, e.g., 1e-12.

        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size) # ASSUMED: A01
        self.token_type_embeddings = nn.Embedding(self.type_vocab_size, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) # Eq. (Layer Normalization)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings # Eq. (Embedding Summation)
        embeddings = self.LayerNorm(embeddings) # Eq. (Layer Normalization)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = TODO_HIDDEN_SIZE # INFERRED: From BERT base config, e.g., 768.
        self.num_attention_heads = TODO_NUM_ATTENTION_HEADS # INFERRED: From BERT base config, e.g., 12.
        self.attention_probs_dropout_prob = TODO_ATTENTION_PROBS_DROPOUT_PROB # INFERRED: From BERT config, e.g., 0.1.
        self.attention_head_size = self.hidden_size // self.num_attention_heads # INFERRED: Standard calculation.
        self.all_head_size = self.num_attention_heads * self.attention_head_size # INFERRED: Standard calculation.

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Eq. (Scaled Dot-Product Attention - part 1)
        attention_scores = attention_scores / Math.sqrt(self.attention_head_size) # Eq. (Scaled Dot-Product Attention - part 2)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask # Eq. (Masking for attention scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # Eq. (Softmax)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # Eq. (Scaled Dot-Product Attention - part 3)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = TODO_HIDDEN_SIZE # INFERRED: From BERT base config, e.g., 768.
        self.hidden_dropout_prob = TODO_HIDDEN_DROPOUT_PROB # INFERRED: From BERT config, e.g., 0.1.
        self.layer_norm_eps = TODO_LAYER_NORM_EPS # INFERRED: From BERT config, e.g., 1e-12.

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) # Eq. (Layer Normalization)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # Eq. (Residual Connection + Layer Normalization)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = TODO_HIDDEN_SIZE # INFERRED: From BERT base config, e.g., 768.
        self.intermediate_size = TODO_INTERMEDIATE_SIZE # INFERRED: From BERT base config, e.g., 3072.
        self.hidden_act = TODO_HIDDEN_ACT # INFERRED: From BERT config, e.g., 'gelu'.

        self.dense = nn.Linear(self.hidden_size, self.intermediate_size)
        if self.hidden_act == "gelu":
            self.intermediate_act_fn = gelu
        elif self.hidden_act == "relu":
            self.intermediate_act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {self.hidden_act}")

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states) # Eq. (Activation Function)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = TODO_HIDDEN_SIZE # INFERRED: From BERT base config, e.g., 768.
        self.intermediate_size = TODO_INTERMEDIATE_SIZE # INFERRED: From BERT base config, e.g., 3072.
        self.hidden_dropout_prob = TODO_HIDDEN_DROPOUT_PROB # INFERRED: From BERT config, e.g., 0.1.
        self.layer_norm_eps = TODO_LAYER_NORM_EPS # INFERRED: From BERT config, e.g., 1e-12.

        self.dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) # Eq. (Layer Normalization)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # Eq. (Residual Connection + Layer Normalization)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_hidden_layers = TODO_NUM_HIDDEN_LAYERS # INFERRED: From BERT base config, e.g., 12.
        self.layer = nn.ModuleList([BertLayer() for _ in range(self.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        # The paper states "The final hidden state T_i for each input token i"
        # This implies the output of the last layer.
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states # This is T_i

class BertModel(nn.Module):
    """
    The bare BERT Model transformer outputting raw hidden-states (T_i) without any specific head on top.
    """
    def __init__(self):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoder()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are (batch_size, 1, 1, to_seq_length)
        # So we can broadcast to (batch_size, num_heads, from_seq_length, to_seq_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # This effectively masks out attention to padded tokens by setting their scores to a very small number.
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # Eq. (Masking for attention scores)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        # The final hidden states (T_i) are the output of the BertEncoder
        final_hidden_states = encoder_outputs
        return final_hidden_states
