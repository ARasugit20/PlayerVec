"""Tests for model tensor behavior and embedding sanity checks."""

import pytest
import torch
import numpy as np
from pathlib import Path
from model.autoencoder import PlayerEncoder, PlayerDecoder, PlayerAutoencoder


class TestPlayerEncoder:
    """Test encoder tensor operations."""

    def test_encoder_output_shape(self):
        """Encoder output has correct embedding dimension."""
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        x = torch.randn(16, 30)
        output = encoder(x)
        assert output.shape == (16, 32)

    def test_encoder_batch_dimension(self):
        """Encoder preserves batch dimension."""
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        for batch_size in [1, 8, 64]:
            x = torch.randn(batch_size, 30)
            output = encoder(x)
            assert output.shape[0] == batch_size

    def test_encoder_device_consistency(self):
        """Encoder works with different devices."""
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        x = torch.randn(4, 30)
        output = encoder(x)
        assert output.device.type == x.device.type

    def test_encoder_dtype_float32(self):
        """Encoder preserves float32 dtype."""
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        x = torch.randn(4, 30, dtype=torch.float32)
        output = encoder(x)
        assert output.dtype == torch.float32

    def test_encoder_non_zero_output(self):
        """Encoder doesn't always output zeros."""
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        x = torch.randn(10, 30)
        output = encoder(x)
        assert (output != 0).any()

    def test_encoder_deterministic(self):
        """Same input produces same output."""
        torch.manual_seed(42)
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        x = torch.randn(4, 30)
        
        torch.manual_seed(0)
        out1 = encoder(x.clone())
        out2 = encoder(x.clone())
        
        torch.testing.assert_close(out1, out2)


class TestPlayerDecoder:
    """Test decoder tensor operations."""

    def test_decoder_reconstruction_shape(self):
        """Decoder reconstructs to original input shape."""
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        z = torch.randn(16, 32)
        output = decoder(z)
        assert output.shape == (16, 30)

    def test_decoder_batch_dimension(self):
        """Decoder preserves batch dimension."""
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        for batch_size in [1, 8, 64]:
            z = torch.randn(batch_size, 32)
            output = decoder(z)
            assert output.shape[0] == batch_size

    def test_decoder_dtype_float32(self):
        """Decoder preserves float32 dtype."""
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        z = torch.randn(4, 32, dtype=torch.float32)
        output = decoder(z)
        assert output.dtype == torch.float32

    def test_decoder_non_zero_output(self):
        """Decoder produces non-trivial output."""
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        z = torch.randn(10, 32)
        output = decoder(z)
        assert (output != 0).any()


class TestPlayerAutoencoder:
    """Test autoencoder round-trip and loss behavior."""

    def test_autoencoder_forward_shape(self):
        """Autoencoder forward pass shape consistency."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(8, 30)
        reconstructed, _ = ae.forward(x)
        assert reconstructed.shape == x.shape

    def test_autoencoder_embedding_extraction(self):
        """Autoencoder correctly extracts embedding."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(8, 30)
        _, embedding = ae.forward(x)
        assert embedding.shape == (8, 32)

    def test_autoencoder_reconstruction_difference(self):
        """Reconstruction differs from input (before training)."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(8, 30)
        reconstructed, _ = ae.forward(x)
        mse = torch.nn.functional.mse_loss(x, reconstructed)
        assert mse > 0  # Untrained model won't reconstruct perfectly

    def test_autoencoder_embedding_normalized(self):
        """Embeddings are L2-normalized."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(16, 30)
        _, embedding = ae.forward(x)
        norms = torch.norm(embedding, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), atol=1e-6, rtol=1e-6)

    def test_autoencoder_gradient_flow(self):
        """Gradients flow through autoencoder."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(8, 30, requires_grad=True)
        reconstructed, _ = ae.forward(x)
        loss = reconstructed.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


class TestEmbeddingProperties:
    """Test embedding vector properties."""

    def test_embedding_finite_values(self):
        """Embeddings contain only finite values."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(10, 30)
        _, embedding = ae.forward(x)
        assert torch.isfinite(embedding).all()

    def test_embedding_no_nans(self):
        """Embeddings contain no NaN values."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(10, 30)
        _, embedding = ae.forward(x)
        assert not torch.isnan(embedding).any()

    def test_embedding_cosine_similarity(self):
        """Cosine similarity between identical inputs is ~1.0."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        x = torch.randn(1, 30)
        _, z1 = ae.forward(x)
        _, z2 = ae.forward(x)
        cosine_sim = torch.nn.functional.cosine_similarity(z1, z2)
        assert cosine_sim.item() > 0.99

    def test_embedding_dimension_mismatch(self):
        """Specifying embedding dim is respected."""
        for dim in [16, 32, 64]:
            ae = PlayerAutoencoder(input_dim=30, embedding_dim=dim)
            x = torch.randn(8, 30)
            _, embedding = ae.forward(x)
            assert embedding.shape[1] == dim
