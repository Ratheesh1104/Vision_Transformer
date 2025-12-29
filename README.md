# Vision Transformer (ViT)

A simple implementation and explanation of the **Vision Transformer (ViT)** model for image classification using deep learning.

## Overview

Vision Transformer (ViT) applies the Transformer architecture, originally designed for NLP, to computer vision tasks.  
Instead of using convolutions, ViT splits an image into fixed-size patches, embeds them, and processes them using self-attention.

## How It Works

1. **Image Patching**  
   The input image is divided into fixed-size patches (e.g., 16×16).

2. **Patch Embedding**  
   Each patch is flattened and projected into a vector using a linear layer.

3. **Positional Encoding**  
   Positional embeddings are added to retain spatial information.

4. **Transformer Encoder**  
   Multiple encoder layers with:
   - Multi-Head Self Attention
   - Feed Forward Network
   - Layer Normalization & Residual Connections

5. **Classification Head**  
   A special `[CLS]` token is used to represent the whole image and is passed to a classifier.
