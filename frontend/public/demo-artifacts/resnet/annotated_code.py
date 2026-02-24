# Component: Batch Normalization
# Provenance: paper-stated
# Assumption: Implementation for 2D inputs (N, C, H, W) as it's common in vision tasks. If a different input dimension was intended (e.g., 1D or 3D), the mean/var dimensions would change.
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBatchNorm2d(nn.Module):
    """
    Custom implementation of Batch Normalization for 2D inputs (e.g., images).
    This module applies Batch Normalization over a mini-batch of 2D inputs.
    """
    # ASSUMED: Implementation for 2D inputs (N, C, H, W) as it's common in vision tasks.
    # If a different input dimension was intended (e.g., 1D or 3D), the mean/var dimensions would change.

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        """
        Initializes the Batch Normalization layer.

        Args:
            num_features (int): Number of features (channels) in the input.
            eps (float): A small value added to the variance to avoid division by zero.
                         # INFERRED: Standard default value in PyTorch's BatchNorm.
            momentum (float): The value used for the running_mean and running_var computation.
                              # INFERRED: Standard default value in PyTorch's BatchNorm.
            affine (bool): If True, this module has learnable affine parameters (gamma and beta).
                           # INFERRED: Standard practice to include learnable scale (gamma) and shift (beta).
            track_running_stats (bool): If True, tracks the running mean and variance.
                                        # INFERRED: Standard practice to track running statistics for inference.
        """
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            # Learnable scale parameter (gamma)
            self.weight = nn.Parameter(torch.ones(num_features))
            # Learnable shift parameter (beta)
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            # Buffers for running statistics, not updated by backprop
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            # Counter for number of batches processed, used for unbiased updates in some cases
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Batch Normalization.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after Batch Normalization.
        """
        # Determine whether to use batch statistics or running statistics
        if self.training and self.track_running_stats:
            # Training mode with running stats tracking: calculate batch stats and update running stats
            batch_mean = input.mean([0, 2, 3]) # Eq. (1)
            batch_var = input.var([0, 2, 3], unbiased=True) # Eq. (2)

            # Update running mean and variance using exponential moving average
            # INFERRED: Standard update rule for running statistics in PyTorch's BatchNorm.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            self.num_batches_tracked += 1

            current_mean = batch_mean
            current_var = batch_var
        elif not self.training and self.track_running_stats:
            # Evaluation mode with running stats tracking: use tracked running stats
            current_mean = self.running_mean
            current_var = self.running_var
        elif not self.track_running_stats:
            # If not tracking running stats (e.g., like InstanceNorm behavior), always use batch stats
            # INFERRED: This behavior aligns with PyTorch's BatchNorm when track_running_stats=False.
            batch_mean = input.mean([0, 2, 3])
            batch_var = input.var([0, 2, 3], unbiased=True)
            current_mean = batch_mean
            current_var = batch_var
        else:
            # This case should ideally not be reached with the above conditions.
            # It would imply self.training is True but track_running_stats is False, which is covered by the last 'elif' branch.
            # For robustness, could raise an error or default to batch stats.
            raise RuntimeError("Unexpected state in BatchNorm forward pass.")


        # Normalize input: x_hat = (x - mu_B) / sqrt(sigma_B^2 + epsilon)
        # The current_mean and current_var are (C,) tensors.
        # We need to reshape them to (1, C, 1, 1) for broadcasting across (N, C, H, W) input.
        normalized_input = (input - current_mean.view(1, -1, 1, 1)) / torch.sqrt(current_var.view(1, -1, 1, 1) + self.eps) # Eq. (3)

        # Scale and shift: y = gamma * x_hat + beta
        if self.affine:
            # weight (gamma) and bias (beta) are (C,) tensors.
            # Reshape to (1, C, 1, 1) for broadcasting.
            output = self.weight.view(1, -1, 1, 1) * normalized_input + self.bias.view(1, -1, 1, 1) # Eq. (4)
        else:
            output = normalized_input

        return output

# Component: Convolutional Layer
# Provenance: inferred
# Assumption: The default value for `bias` is set to `False` when `None` is passed, based on ambiguity resolution A04, which states that convolutional layers followed by Batch Normalization should not include bias terms. The paper implies BN is used after each convolution.
# Assumption: The 'n_in' mentioned in ambiguity resolution A03 for weight initialization is interpreted as 'fan_in' for convolutional layers, which is `in_channels * kernel_height * kernel_width`. PyTorch's `kaiming_normal_` with `mode='fan_in'` and `nonlinearity='relu'` is used to implement this He initialization.
# Assumption: Bias terms, if present (i.e., if `bias=True` was explicitly passed), are initialized to zero.
import torch
import torch.nn as nn
import torch.nn.init as init

class ConvolutionalLayer(nn.Module):
    """
    Implementation for a Convolutional Layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = None):
        """
        Initializes the ConvolutionalLayer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output.
                                   Defaults to None, which infers False based on A04.
        """
        super().__init__()

        # INFERRED: Default stride, padding, dilation, groups are standard for nn.Conv2d.
        # ASSUMED: If bias is not explicitly provided, it defaults to False based on A04.
        # A04: "Do not include bias terms in convolutional or fully-connected layers that are followed by a Batch Normalization layer. The paper states BN is used after each convolution."
        if bias is None:
            bias = False # INFERRED: Based on A04, BN is used after each convolution, so bias is typically False.

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # A03: Weight Initialization
        # "Implement the weight initialization from [13]. For a given layer, the weights should be drawn from a zero-mean Gaussian distribution with a standard deviation of sqrt(2 / n_in), where n_in is the number of input units to the layer."
        # For a convolutional layer, 'n_in' (or 'fan_in') is typically `in_channels * kernel_height * kernel_width`.
        # PyTorch's `kaiming_normal_` with `mode='fan_in'` and `nonlinearity='relu'` correctly implements this
        # for ReLU activations, which is a common pairing for He initialization.
        init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu') # Eq. (N) - Refers to [13] via A03
        # ASSUMED: The 'n_in' in A03 refers to 'fan_in' for convolutional layers, which is in_channels * kernel_height * kernel_width.
        # INFERRED: PyTorch's `kaiming_normal_` with `mode='fan_in'` and `nonlinearity='relu'` correctly implements this.

        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0) # ASSUMED: Bias terms, if present, are initialized to zero.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution.
        """
        return self.conv(x)

# Component: Data Augmentation
# Provenance: inferred
# Assumption: AlexNetColorAugmentation: eigvecs and eigvals are pre-computed from ImageNet training set. These values are not provided in the prompt and cannot be computed here.
# Assumption: AlexNetColorAugmentation: Input tensor to __call__ is already normalized to [0, 1] by transforms.ToTensor().
# Assumption: get_data_augmentation_transforms: initial_resize_size (e.g., 256) is a common practice for ImageNet for both training (implicitly handled by RandomResizedCrop) and validation/test (explicitly used for Resize).
# Assumption: get_data_augmentation_transforms: pca_eigvecs and pca_eigvals are pre-computed from ImageNet training set.
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image # PIL is used by torchvision transforms, so keep it for context

class AlexNetColorAugmentation(object):
    """
    Implements the color augmentation described in the AlexNet paper [21].
    Performs PCA on RGB pixel values and adds multiples of principal components.
    This transform expects a torch.Tensor input of shape (C, H, W) with pixel values
    in the range [0, 1].
    """
    def __init__(self, eigvecs, eigvals, alpha_std=0.1):
        """
        Args:
            eigvecs (torch.Tensor or np.ndarray): Pre-computed eigenvectors from PCA on ImageNet RGB pixels.
                                                  Shape: (3, 3)
            eigvals (torch.Tensor or np.ndarray): Pre-computed eigenvalues from PCA on ImageNet RGB pixels.
                                                  Shape: (3,)
            alpha_std (float): Standard deviation for the Gaussian random variable.
                               INFERRED: 0.1 based on AlexNet paper [21] referenced by A02.
        """
        # ASSUMED: eigvecs and eigvals are pre-computed from ImageNet training set.
        # These values are not provided in the prompt and cannot be computed here.
        if not isinstance(eigvecs, torch.Tensor):
            self.eigvecs = torch.from_numpy(eigvecs).float()
        else:
            self.eigvecs = eigvecs.float()

        if not isinstance(eigvals, torch.Tensor):
            self.eigvals = torch.from_numpy(eigvals).float()
        else:
            self.eigvals = eigvals.float()

        if self.eigvecs.shape != (3, 3) or self.eigvals.shape != (3,):
            raise ValueError("eigvecs must be (3, 3) and eigvals must be (3,)")

        self.alpha_std = alpha_std

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (torch.Tensor): Image to be color augmented.
                                       Expected shape: (C, H, W) and pixel values in [0, 1].
        Returns:
            torch.Tensor: Color augmented image, shape (C, H, W), values clamped to [0, 1].
        """
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input to AlexNetColorAugmentation must be a torch.Tensor.")
        if img_tensor.dim() != 3 or img_tensor.shape[0] != 3:
            raise ValueError("Input Tensor must be of shape (C, H, W) with C=3.")
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.float()
        # ASSUMED: Input tensor is already normalized to [0, 1] by transforms.ToTensor()

        # Convert to (H, W, C) for easier pixel manipulation
        img_tensor_hwc = img_tensor.permute(1, 2, 0).clone() # (H, W, C)

        # Reshape image to (N_pixels, 3) for PCA application
        original_shape = img_tensor_hwc.shape
        img_flat = img_tensor_hwc.view(-1, 3) # (H*W, 3)

        # Generate random variables alpha_i from N(0, alpha_std)
        # INFERRED: Standard deviation for Gaussian noise is 0.1 based on AlexNet paper [21] referenced by A02.
        alphas = torch.randn(3, device=img_tensor.device) * self.alpha_std # (3,)

        # Calculate the perturbation vector p = E * (alpha * lambda)
        # where E are eigenvectors, lambda are eigenvalues
        # Eq. (A02 description)
        perturbation = torch.matmul(self.eigvecs.to(img_tensor.device), alphas * self.eigvals.to(img_tensor.device)) # (3, 3) * (3,) -> (3,)

        # Add perturbation to each pixel
        # Clamp to [0, 1] range after adding perturbation
        # Eq. (A02 description)
        augmented_img_flat = img_flat + perturbation
        augmented_img_flat = torch.clamp(augmented_img_flat, 0.0, 1.0)

        # Reshape back to original image dimensions (H, W, C)
        augmented_img_hwc = augmented_img_flat.view(original_shape)

        # Convert back to (C, H, W) for consistency with torchvision transforms
        return augmented_img_hwc.permute(2, 0, 1)

def get_data_augmentation_transforms(
    is_train: bool,
    image_size: int = 224,
    initial_resize_size: int = 256, # ASSUMED: Common practice for ImageNet.
    normalize_mean: list = None,
    normalize_std: list = None,
    pca_eigvecs: np.ndarray = None, # Placeholder for pre-computed PCA components
    pca_eigvals: np.ndarray = None, # Placeholder for pre-computed PCA components
    color_augmentation_std: float = 0.1 # INFERRED: 0.1 based on AlexNet paper [21] referenced by A02.
):
    """
    Generates torchvision transforms for data augmentation.

    Args:
        is_train (bool): If True, applies training augmentations (random crop, flip, color jitter).
                         If False, applies validation/test augmentations (center crop).
        image_size (int): The final size of the image after cropping (e.g., 224 for ImageNet).
        initial_resize_size (int): For validation/test, the shortest side of the image is resized to this.
                                   For training, RandomResizedCrop handles resizing internally.
                                   ASSUMED: 256 for ImageNet, as per common practice.
        normalize_mean (list): Mean values for image normalization (e.g., [0.485, 0.456, 0.406] for ImageNet).
                               If None, normalization is skipped.
        normalize_std (list): Standard deviation values for image normalization (e.g., [0.229, 0.224, 0.225] for ImageNet).
                              If None, normalization is skipped.
        pca_eigvecs (np.ndarray): Pre-computed eigenvectors for AlexNet-style color augmentation.
                                  Shape (3, 3). Required if color augmentation is desired.
                                  ASSUMED: Pre-computed from ImageNet training set.
        pca_eigvals (np.ndarray): Pre-computed eigenvalues for AlexNet-style color augmentation.
                                  Shape (3,). Required if color augmentation is desired.
                                  ASSUMED: Pre-computed from ImageNet training set.
        color_augmentation_std (float): Standard deviation for the Gaussian random variable
                                        used in AlexNet-style color augmentation.
                                        INFERRED: 0.1 based on AlexNet paper [21] referenced by A02.

    Returns:
        torchvision.transforms.Compose: A composition of data augmentation transforms.
    """
    transform_list = []

    if is_train:
        # "randomly crop a 224x224 region from an image or its horizontal flip"
        # RandomResizedCrop handles both resizing and cropping to the target size.
        transform_list.append(transforms.RandomResizedCrop(image_size))
        transform_list.append(transforms.RandomHorizontalFlip())

        # A02: Implement AlexNet-style color augmentation
        if pca_eigvecs is not None and pca_eigvals is not None:
            transform_list.append(transforms.ToTensor()) # Convert to Tensor (C, H, W) for custom transform
            transform_list.append(AlexNetColorAugmentation(pca_eigvecs, pca_eigvals, color_augmentation_std))
            # AlexNetColorAugmentation now guarantees (C, H, W) output.
        else:
            # If no PCA color augmentation, convert to Tensor here
            transform_list.append(transforms.ToTensor())

    else:
        # For validation/testing
        # Resize shortest side to initial_resize_size, then center crop to image_size.
        transform_list.append(transforms.Resize(initial_resize_size)) # ASSUMED: Resize shortest side to 256 for validation/test
        transform_list.append(transforms.CenterCrop(image_size))
        transform_list.append(transforms.ToTensor())

    if normalize_mean is not None and normalize_std is not None:
        transform_list.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

    return transforms.Compose(transform_list)
