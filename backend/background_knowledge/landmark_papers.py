from __future__ import annotations


LANDMARK_PAPERS: dict[str, str] = {
    "attention is all you need": (
        "Transformer architecture replaces recurrence with self-attention, enabling parallel sequence modeling."
    ),
    "resnet": "Residual connections stabilize deep training by adding identity shortcuts.",
    "batch normalization": "BatchNorm normalizes activations per batch for optimization stability.",
    "adam": "Adam combines adaptive learning rates with momentum-like running averages.",
    "bert": "BERT pretrains bidirectional Transformer encoders with masked language modeling.",
    "gpt-2": "GPT-2 scales decoder-only autoregressive Transformer language modeling.",
    "vision transformer": "ViT tokenizes image patches and processes them with Transformer encoders.",
    "ddpm": "Diffusion models learn to reverse a noising process for generative sampling.",
}

