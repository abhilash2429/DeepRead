# Transformer Architecture Summary

## 1. What This Paper Actually Does
Existing models for tasks like machine translation, which process sequences of words, faced two main challenges: they were either slow because they had to process words one after another, or they struggled to understand how distant words in a long sentence related to each other. This paper introduces the Transformer, a novel model that completely abandons these traditional approaches. Instead, it relies entirely on a mechanism called 'attention,' which allows the model to simultaneously look at all other words in a sentence and decide which ones are most relevant to understanding the current word. The Transformer uses this attention mechanism in both an 'encoder' that processes the input sentence and a 'decoder' that generates the output sentence, ensuring it still understands the order of words by adding special 'positional encodings.' This design is revolutionary because it allows the model to process all words in a sentence simultaneously, dramatically speeding up training compared to older methods. Crucially, it also makes it much easier for the model to identify connections between words, no matter how far apart they are in a sentence. As a result, the Transformer achieved state-of-the-art performance on complex tasks like machine translation, setting a new standard for how sequence data is processed in AI.

## 2. The Mechanism
### Briefing Section 2: The Mechanism

The Transformer model processes sequence-to-sequence tasks using an encoder-decoder architecture, as illustrated in Figure 1. The encoder maps an input sequence into a continuous representation, which the decoder then uses to generate an output sequence one token at a time. The entire process is built upon attention mechanisms, eschewing recurrence and convolutions.

#### 2.1. Input and Positional Encoding

The process begins by converting input tokens into high-dimensional vectors.

1.  **Token Embedding**: Both the source and target sequences are first passed through separate embedding layers to convert each token ID into a vector of dimension `d_model` = 512.
2.  **Positional Encoding**: Since the model contains no recurrent or convolutional layers, it has no inherent sense of token order. To provide this crucial information, positional encodings are added to the input embeddings. These encodings are fixed, non-learned vectors calculated using sine and cosine functions of different frequencies:

    > PE<sub>(pos, 2i)</sub> = sin(pos / 10000<sup>2i / d_model</sup>)
    > PE<sub>(pos, 2i+1)</sub> = cos(pos / 10000<sup>2i / d_model</sup>)

    Here, `pos` is the position of the token in the sequence, and `i` is the dimension within the embedding vector. This sinusoidal approach allows the model to learn relative positional relationships, as the encoding for any position `pos + k` can be represented as a linear function of the encoding for position `pos` (Section 3.5).

#### 2.2. The Encoder Stack

The encoder's role is to build a rich, context-aware representation of the entire input sequence. It consists of a stack of `N`=6 identical layers. The output of the final encoder layer is a sequence of vectors, one for each input token, which is then passed to every layer in the decoder. Each encoder layer has two sub-layers.

##### 2.2.1. Multi-Head Self-Attention

The first sub-layer allows each token in the input sequence to look at all other tokens in the sequence to better inform its own representation. This is the core mechanism for understanding context. It is built from a fundamental unit called **Scaled Dot-Product Attention**, depicted in Figure 2 (left).

The attention score is calculated as:

> Attention(Q, K, V) = softmax( (QK<sup>T</sup>) / √d<sub>k</sub> ) V --- (Eq. 1)

-   **Prerequisite**: The input to this layer is a sequence of vectors. For each vector, three new vectors are created through linear projections: a **Query (Q)**, a **Key (K)**, and a **Value (V)**. In self-attention, Q, K, and V are all derived from the same input sequence (the output of the previous layer).
-   **Step 1: Score Calculation**: The dot product of a query vector `Q` with all key vectors `K` is computed to produce a raw similarity score between them.
-   **Step 2: Scaling**: This score is scaled down by dividing by √d<sub>k</sub>, where `d_k` is the dimension of the key vectors (here, `d_k`=64). This scaling is necessary because for large values of `d_k`, the dot products can become very large, pushing the softmax function into regions with vanishingly small gradients, which would impede learning (Section 3.2.1).
-   **Step 3: Weighting**: A softmax function is applied to the scaled scores to obtain attention weights, which are positive and sum to one. These weights determine how much focus to place on each input token when encoding the current token.
-   **Step 4: Output**: The final output for the query is a weighted sum of all the value vectors `V`, using the computed attention weights.

Instead of performing a single attention function, the model employs **Multi-Head Attention** (Figure 2, right). This involves projecting the Q, K, and V vectors `h`=8 times with different, learned linear projections. Each of these projected versions of Q, K, and V is fed into a separate attention "head." The outputs of these 8 heads are then concatenated and projected again with another learned linear transformation to produce the final output. This allows the model to jointly attend to information from different representation subspaces at different positions, enriching its ability to capture complex relationships (Section 3.2.2).

##### 2.2.2. Position-wise Feed-Forward Network

The second sub-layer is a simple, fully connected feed-forward network (FFN) applied to each position's vector independently. It consists of two linear transformations with a ReLU activation in between:

> FFN(x) = max(0, xW<sub>1</sub> + b<sub>1</sub>)W<sub>2</sub> + b<sub>2</sub> --- (Eq. 2)

-   Here, `x` is the output from the attention sub-layer. `W_1` and `b_1` are the weight matrix and bias for the first linear transformation (which expands the dimension from `d_model`=512 to `d_ff`=2048), and `W_2` and `b_2` are for the second (which projects it back to `d_model`=512). This sub-layer provides non-linearity and further transforms the representations.

##### 2.2.3. Residual Connections and Normalization

Each of the two sub-layers (Multi-Head Attention and FFN) in a layer has a residual connection around it, followed by layer normalization (Section 3.1). The output of each sub-layer is `LayerNorm(x + Sublayer(x))`. This is a critical component that allows for the training of a deep stack of `N` layers by preventing gradients from vanishing and stabilizing the learning process.

#### 2.3. The Decoder Stack

The decoder's role is to generate the output sequence one token at a time, using the encoder's output as context. It also consists of a stack of `N`=6 identical layers. For each step in the output sequence, the decoder takes the previously generated tokens as input.

-   **Prerequisite**: The decoder input is the target sequence, "shifted right." This means that for predicting the token at position `i`, the decoder is only given the ground-truth tokens from positions 1 to `i-1`. This preserves the **auto-regressive** property, ensuring predictions are based only on past information.

Each decoder layer has three sub-layers.

##### 2.3.1. Masked Multi-Head Self-Attention

This sub-layer is nearly identical to the self-attention mechanism in the encoder. However, it is modified to prevent positions from attending to subsequent positions. This is achieved by applying a "look-ahead mask" inside the Scaled Dot-Product Attention (the "Mask (opt.)" step in Figure 2). Before the softmax step, the mask sets all values corresponding to future positions to negative infinity, effectively zeroing out their attention weights. This is necessary to maintain the auto-regressive property during training.

##### 2.3.2. Multi-Head Cross-Attention

This second sub-layer is what connects the encoder and decoder. It performs multi-head attention, but its inputs are different:
-   The **Queries (Q)** come from the output of the previous decoder sub-layer (the masked self-attention).
-   The **Keys (K)** and **Values (V)** come from the output of the final layer of the encoder stack.

This mechanism allows every position in the decoder to attend over all positions in the input sequence, enabling it to weigh the importance of different parts of the source sentence when generating the next target token.

##### 2.3.3. Position-wise Feed-Forward Network

This third sub-layer is identical in structure and function to the FFN in the encoder layer.

As in the encoder, each of these three sub-layers is wrapped with a residual connection and layer normalization.

#### 2.4. Final Output Generation

After the final decoder layer produces its output vectors, a final linear transformation projects these vectors into a much larger vector with dimensions equal to the size of the target vocabulary. A softmax function is then applied to this vector to convert the scores (logits) into a probability distribution. The token with the highest probability is chosen as the output for that time step.

## 3. What You Need To Already Know
Here are the foundational concepts required to understand the "Attention Is All You Need" paper, presented in a dependency-ordered list:

### 1. Recurrent Neural Networks (RNNs)

1)  **Problem**
    How can a neural network process sequential data (like sentences) where the order of elements is crucial and the input length can vary? Traditional feed-forward networks struggle with variable-length inputs and maintaining context over time.

2)  **Solution**
    RNNs introduce a 'memory' or 'hidden state' that is updated at each step of the sequence. The output at a given step `t` is a function of the input at step `t` and the hidden state from step `t-1`. This recurrent loop allows information to persist through the sequence, capturing temporal dependencies. Variants like LSTMs and GRUs address vanishing/exploding gradients in long sequences.

3)  **Usage in this paper**
    The Transformer is explicitly designed to replace RNNs. The paper argues that the inherently sequential nature of RNNs (where `h_t` depends on `h_{t-1}`) is a major bottleneck for parallelization during training and limits their efficiency for very long sequences. The Transformer's attention-only architecture solves these limitations.

### 2. Encoder-Decoder Architecture

1)  **Problem**
    How can a model handle sequence-to-sequence tasks (like machine translation or text summarization) where the input and output sequences can have different lengths, vocabularies, and grammatical structures?

2)  **Solution**
    This architecture is split into two main parts: an 'encoder' that reads the entire input sequence and transforms it into a rich, fixed-size context representation (or a sequence of context vectors), and a 'decoder' that takes this context and generates the output sequence one element at a time, conditioned on previously generated elements.

3)  **Usage in this paper**
    The Transformer follows this high-level encoder-decoder structure. The left side of Figure 1 in the paper illustrates the encoder stack, which processes the input sentence, and the right side shows the decoder stack, which generates the translated sentence.

### 3. Attention Mechanisms

1)  **Problem**
    In the basic encoder-decoder architecture (especially with RNNs), the entire meaning of a long input sequence is often compressed into a single, fixed-size context vector by the encoder. This creates an information bottleneck, making it difficult for the decoder to access specific, relevant details from the input when generating later parts of the output sequence.

2)  **Solution**
    Attention allows the decoder, at each step of generating an output, to dynamically look back at all parts of the encoder's output (or its own previous outputs). It computes a set of 'attention weights' to determine which input parts are most relevant for the current output step and creates a weighted average of these parts as a dynamic context vector. This provides a flexible way to access information without a fixed-size bottleneck.

3)  **Usage in this paper**
    Attention is the core building block of the Transformer, replacing recurrence and convolutions entirely. It is used in three distinct ways:
    *   **Self-attention in the encoder:** Input tokens attend to other input tokens to build richer representations.
    *   **Masked self-attention in the decoder:** Output tokens attend to previous output tokens to maintain the auto-regressive property during generation.
    *   **Cross-attention between encoder and decoder:** The decoder attends to the encoder's output, mimicking traditional attention mechanisms to align input and output.

### 4. Residual Connections (Skip Connections)

1)  **Problem**
    As neural networks get deeper (i.e., have many layers), they become very difficult to train effectively. A common issue is the 'vanishing gradient' problem, where gradients shrink exponentially as they are backpropagated through many layers, preventing weights in early layers from updating and learning.

2)  **Solution**
    Residual (or 'skip') connections add the input of a layer (or a block of layers) directly to its output. Mathematically, if `F(x)` is the output of a layer, the residual connection makes the output `x + F(x)`. This creates a direct path for the gradient to flow through the network, mitigating the vanishing gradient problem and allowing for the training of much deeper models without significant performance degradation.

3)  **Usage in this paper**
    Residual connections are employed extensively throughout the Transformer. They are used around each of the two sub-layers (the multi-head attention sub-layer and the position-wise feed-forward network sub-layer) in every encoder and decoder layer. The paper specifies the operation as `LayerNorm(x + Sublayer(x))`.

### 5. Layer Normalization

1)  **Problem**
    During training, the distribution of each layer's inputs changes as the parameters of the previous layers change. This phenomenon, called 'internal covariate shift', can slow down training, make it unstable, and require careful initialization and lower learning rates.

2)  **Solution**
    Layer Normalization stabilizes the distributions by normalizing the inputs to a layer. For each training example and for each layer, it computes the mean and variance across all features (or hidden units) for that *single example* and uses them to rescale the inputs. This helps to speed up and stabilize training, making deep networks less sensitive to initialization and more robust.

3)  **Usage in this paper**
    Layer Normalization is applied after each residual connection in both the encoder and decoder layers. It is a critical component for stabilizing the training of the deep Transformer architecture, especially given its post-normalization placement (`LayerNorm(x + Sublayer(x))`).

### 6. Label Smoothing

1)  **Problem**
    When training a classification model with a softmax output and one-hot labels (e.g., `[0, 1, 0]`), the model is encouraged to make its predictions extremely confident (pushing one logit to a very high positive value and others to very low negative values). This can lead to over-fitting, poor calibration (overestimating probabilities), and reduced generalization ability, especially if the training data contains noise or mislabeled examples.

2)  **Solution**
    Label smoothing replaces the hard 0 and 1 targets with soft targets. For example, a target of `[0, 1, 0]` might be changed to `[0.05, 0.9, 0.05]` (where `0.05` is `epsilon_ls / (num_classes - 1)` and `0.9` is `1 - epsilon_ls`). This discourages the model from becoming overconfident, forces it to learn a more robust internal representation, and improves generalization.

3)  **Usage in this paper**
    Label smoothing with a value of `epsilon_ls = 0.1` is used as a regularization technique during training of the Transformer. The paper notes that while this technique might slightly increase the perplexity (a measure of how well the model predicts a sample), it consistently improves accuracy and the BLEU score (a common metric for machine translation quality).
