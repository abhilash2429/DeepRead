# BERT Ambiguity Report

## A01: Position Embedding Implementation
- Type: underspecified_architecture
- Section: 3 BERT, Input/Output Representations
- Ambiguous point: The paper states that position embeddings are learned, not the fixed sinusoidal embeddings from the original Transformer paper. However, it does not specify how they are learned, their maximum length, or how the model would handle sequences longer than the 512 tokens seen during pre-training.
- Implementation consequence: If an implementer assumes a fixed maximum length (e.g., 512) for the learned position embeddings, the model will fail on longer sequences at inference time. If the embeddings are not initialized or trained correctly, it could degrade model performance, as positional information is critical for the self-attention mechanism.
- Agent resolution: Assume the position embeddings are a learned lookup table of size (max_sequence_length, hidden_size), e.g., (512, 768). They are trained from scratch along with the rest of the model. For sequences longer than 512, a common strategy is to truncate the input, though this is not specified in the paper.
- Confidence: 0.5

## A02: Tokenizer Vocabulary Generation
- Type: missing_training_detail
- Section: 3 BERT, Input/Output Representations
- Ambiguous point: The paper specifies a 30,000 token WordPiece vocabulary but does not detail its creation process. It's unclear what corpus was used to train the tokenizer, whether it was cased or uncased for pre-training, or what other configuration settings were used.
- Implementation consequence: Using a different vocabulary or tokenizer settings would create a complete mismatch with the released pre-trained weights, making it impossible to replicate the paper's results. The model's performance is highly sensitive to the tokenization scheme.
- Agent resolution: The official BERT implementation released by Google uses a specific vocabulary file (`vocab.txt`). It is standard practice to use this provided file. The base model is uncased, and a separate cased model was also released. The pre-training described for the main results likely used the uncased vocabulary.
- Confidence: 0.5

## A03: NSP Random Sentence Sampling Strategy
- Type: missing_training_detail
- Section: 3.1 Pre-training BERT, Task #2: Next Sentence Prediction (NSP)
- Ambiguous point: For the Next Sentence Prediction task, the paper states that 50% of the time, sentence B is a 'random sentence from the corpus'. It does not specify the sampling strategy: is the random sentence from the same document, a different document, or the entire corpus? Are there constraints on its length?
- Implementation consequence: The difficulty of the NSP task depends heavily on this sampling strategy. If random sentences are always from different documents, the model might learn to solve the task using simple topic differences, rather than learning about coherence and logical flow. This could make the pre-training less effective for downstream NLI tasks.
- Agent resolution: The official implementation samples random sentences from the entire corpus, not just the same document. A sentence is chosen at random, and there are no explicit constraints other than it not being the true next sentence.
- Confidence: 0.5

## A04: SQuAD v2.0 No-Answer Threshold (τ)
- Type: missing_hyperparameter
- Section: 4.3 SQuAD v2.0
- Ambiguous point: For SQuAD v2.0, the model predicts a no-answer response if the score of the best non-null span is not greater than the no-answer span score by a threshold τ. The paper states τ is 'selected on the dev set to maximize F1' but does not provide the value of τ or the search procedure.
- Implementation consequence: Without the value of τ, the exact F1 score on the SQuAD v2.0 dev and test sets cannot be replicated. Different values of τ will produce a different precision/recall trade-off for answerable vs. unanswerable questions, leading to different results.
- Agent resolution: The value of τ must be found by running inference on the development set with the fine-tuned model and searching for the threshold that maximizes the F1 score. A common approach is to iterate through a range of possible score differences observed on the dev set and pick the one that yields the best F1.
- Confidence: 0.5

## A05: TriviaQA Augmentation Details for SQuAD
- Type: missing_training_detail
- Section: 4.2 SQuAD v1.1, Footnote 12
- Ambiguous point: The best SQuAD v1.1 model was first fine-tuned on TriviaQA. The paper provides minimal details on this intermediate step: 'first 400 tokens in documents, that contain at least one of the provided possible answers'. Key details like the learning rate, number of epochs, and batch size for this phase are missing.
- Implementation consequence: The state-of-the-art SQuAD v1.1 results are not reproducible without these crucial hyperparameters. The performance boost from this intermediate training step is significant, and incorrect settings could lead to worse results or negative transfer.
- Agent resolution: Assume the same fine-tuning hyperparameters as the main SQuAD task (e.g., LR=5e-5, Batch=32) and train for a similar number of epochs (e.g., 2-3). This is a reasonable starting point for experimentation.
- Confidence: 0.5

## A06: Details of Random Restarts
- Type: missing_training_detail
- Section: 4.1 GLUE
- Ambiguous point: For BERT_LARGE on GLUE, the authors 'ran several random restarts and selected the best model on the Dev set'. The paper does not specify how many restarts 'several' is, nor what was re-randomized (data shuffling, classifier layer initialization, or both).
- Implementation consequence: The reported GLUE scores for BERT_LARGE might be the result of cherry-picking from an unknown number of runs. This makes it difficult to assess the model's expected performance and variance. An implementer might get a lower score with a single run and incorrectly assume their implementation is flawed.
- Agent resolution: Implementers should be aware that fine-tuning can be unstable. A common practice is to run experiments with 3 to 5 different random seeds and report the mean and standard deviation. To replicate the paper's 'best' score, one would need to run multiple trials and select the best-performing one on the dev set.
- Confidence: 0.5

## A07: Pre-training Loss Weighting
- Type: ambiguous_loss_function
- Section: A.2 Pre-training Procedure
- Ambiguous point: The total pre-training loss is described as 'the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood.' It is not explicitly stated if these two loss components are weighted equally (i.e., weight of 1.0 for each) or if there is some other weighting scheme.
- Implementation consequence: If the two losses are not weighted equally, the model's focus during pre-training would shift. For example, a higher weight on NSP might make the model better at sentence-pair tasks at the expense of token-level understanding. An incorrect implementation of the loss function would lead to a differently optimized pre-trained model.
- Agent resolution: Assume the losses are added with equal weight (1.0). The loss function is `Loss = Loss_MLM + Loss_NSP`.
- Confidence: 0.5
