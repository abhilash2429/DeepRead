# ResNet Ambiguity Report

## A01: ImageNet Learning Rate Schedule Trigger
- Type: missing_hyperparameter
- Section: 3.4. Implementation
- Ambiguous point: The paper states the learning rate for ImageNet training is 'divided by 10 when the error plateaus'.
- Implementation consequence: The term 'plateaus' is not defined. Without knowing the exact metric (training or validation error), the patience (number of epochs/iterations to wait), and the threshold for what constitutes a plateau, the learning rate schedule cannot be reproduced. This will lead to different convergence behavior and final model accuracy.
- Agent resolution: A common implementation is to monitor the validation error and reduce the learning rate if it does not improve for a set number of epochs (e.g., 5-10 epochs). The CIFAR-10 experiments use a fixed iteration-based schedule, which is an alternative. Given the lack of detail, a fixed schedule like the one for CIFAR-10 (e.g., dropping at 300k and 500k iterations) would be a more reproducible choice.
- Confidence: 0.9

## A02: Standard Color Augmentation Details
- Type: missing_training_detail
- Section: 3.4. Implementation
- Ambiguous point: The paper states 'The standard color augmentation in [21] is used.' for ImageNet training.
- Implementation consequence: Reference [21] (Krizhevsky et al., 2012) describes a specific PCA-based color jittering technique. If a developer is unaware of this or implements a different color augmentation (e.g., simple brightness/contrast adjustments), the training data distribution will be different, which can affect the final model's accuracy and robustness.
- Agent resolution: Implement the color augmentation as described in the AlexNet paper [21]. This involves performing PCA on the RGB pixel values of the ImageNet training set, and then for each image, adding multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian distribution.
- Confidence: 1.0

## A03: Weight Initialization Details
- Type: missing_training_detail
- Section: 3.4. Implementation
- Ambiguous point: The paper states 'We initialize the weights as in [13]'.
- Implementation consequence: Reference [13] (He et al., 2015, 'Delving Deep into Rectifiers') introduces a specific initialization method for ReLU networks (often called 'He initialization'). Using a different initialization, like Xavier/Glorot, could lead to slower convergence or prevent very deep networks from converging at all, as it is not specifically designed for ReLU nonlinearities.
- Agent resolution: Implement the weight initialization from [13]. For a given layer, the weights should be drawn from a zero-mean Gaussian distribution with a standard deviation of sqrt(2 / n_in), where n_in is the number of input units to the layer.
- Confidence: 1.0

## A04: Use of Biases in Convolutional/FC Layers
- Type: underspecified_architecture
- Section: 3.2. Identity Mapping by Shortcuts
- Ambiguous point: In Section 3.2, the formula for a residual block is given, and the text notes 'the biases are omitted for simplifying notations'. It is not explicitly stated whether biases are used in the actual implementation.
- Implementation consequence: If biases are added to convolutional layers that are immediately followed by a Batch Normalization layer, the effect of the bias will be cancelled out by the mean subtraction step in BN. Adding them would add useless parameters to the model, slightly increasing memory usage and computation for no benefit. If BN were not present, omitting biases would be a significant architectural change.
- Agent resolution: Do not include bias terms in convolutional or fully-connected layers that are followed by a Batch Normalization layer. The paper states BN is used after each convolution. The final FC layer before the softmax does not have a subsequent BN layer and should include a bias term.
- Confidence: 0.95

## A05: Exact Projection Shortcut Implementation
- Type: underspecified_architecture
- Section: 3.3. Network Architectures
- Ambiguous point: For projection shortcuts (Option B), the paper states they are done by 1x1 convolutions to match dimensions. When crossing feature maps of two sizes, they are performed with a stride of 2. It is not specified if these 1x1 convolutions have a subsequent BN and/or ReLU.
- Implementation consequence: If the projection shortcut path includes BN and ReLU, its statistical properties and non-linearity will be different from a simple linear projection. This could affect how information propagates through the shortcut and impact training dynamics. Most open-source implementations use a 1x1 convolution without any non-linearity or normalization on the shortcut path.
- Agent resolution: The projection shortcut should consist of only a 1x1 convolutional layer with a stride of 2. It should not be followed by Batch Normalization or a ReLU activation. This preserves the shortcut as a linear projection to match dimensions, which is its stated purpose.
- Confidence: 0.9

## A06: Composition of the 6-Model Ensemble
- Type: missing_training_detail
- Section: 4.1. ImageNet Classification
- Ambiguous point: For the best ImageNet result, the paper mentions 'We combine six models of different depth to form an ensemble (only with two 152-layer ones at the time of submitting)'.
- Implementation consequence: The final state-of-the-art result of 3.57% top-5 error cannot be reproduced without knowing the exact architecture of the other four models in the ensemble. The performance of an ensemble is highly dependent on the diversity and individual performance of its constituent models.
- Agent resolution: This result is not reproducible from the paper alone. To create a similar ensemble, one could train one of each of the other architectures presented (e.g., ResNet-34, ResNet-50, ResNet-101) and a sixth model, perhaps another ResNet-152 with a different random seed or a ResNet-101. The final performance will likely differ.
- Confidence: 1.0
