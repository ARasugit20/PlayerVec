"""Tests for model tensor behavior and embedding sanity checks."""

import torch
import torch.nn.functional as functional

from model.autoencoder import PlayerAutoencoder, PlayerDecoder, PlayerEncoder


class TestPlayerEncoder:
    """Test encoder tensor operations."""

    def test_encoder_output_shape(self):
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        encoder.eval()
        x = torch.randn(16, 30)
        output = encoder(x)
        assert output.shape == (16, 32)

    def test_encoder_batch_dimension(self):
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        encoder.eval()
        for batch_size in [1, 8, 64]:
            x = torch.randn(batch_size, 30)
            output = encoder(x)
            assert output.shape[0] == batch_size

    def test_encoder_device_consistency(self):
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        encoder.eval()
        x = torch.randn(4, 30)
        output = encoder(x)
        assert output.device.type == x.device.type

    def test_encoder_dtype_float32(self):
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        encoder.eval()
        x = torch.randn(4, 30, dtype=torch.float32)
        output = encoder(x)
        assert output.dtype == torch.float32

    def test_encoder_non_zero_output(self):
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        encoder.eval()
        x = torch.randn(10, 30)
        output = encoder(x)
        assert (output != 0).any()

    def test_encoder_deterministic(self):
        torch.manual_seed(42)
        encoder = PlayerEncoder(input_dim=30, embedding_dim=32)
        encoder.eval()
        x = torch.randn(4, 30)
        out1 = encoder(x.clone())
        out2 = encoder(x.clone())
        torch.testing.assert_close(out1, out2)


class TestPlayerDecoder:
    """Test decoder tensor operations."""

    def test_decoder_reconstruction_shape(self):
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        decoder.eval()
        z = torch.randn(16, 32)
        output = decoder(z)
        assert output.shape == (16, 30)

    def test_decoder_batch_dimension(self):
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        decoder.eval()
        for batch_size in [1, 8, 64]:
            z = torch.randn(batch_size, 32)
            output = decoder(z)
            assert output.shape[0] == batch_size

    def test_decoder_dtype_float32(self):
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        decoder.eval()
        z = torch.randn(4, 32, dtype=torch.float32)
        output = decoder(z)
        assert output.dtype == torch.float32

    def test_decoder_non_zero_output(self):
        decoder = PlayerDecoder(input_dim=30, embedding_dim=32)
        decoder.eval()
        z = torch.randn(10, 32)
        output = decoder(z)
        assert (output != 0).any()


class TestPlayerAutoencoder:
    """Test autoencoder round-trip and loss behavior."""

    def test_autoencoder_forward_shape(self):
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(8, 30)
        reconstructed, _ = ae.forward(x)
        assert reconstructed.shape == x.shape

    def test_autoencoder_embedding_extraction(self):
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(8, 30)
        _, embedding = ae.forward(x)
        assert embedding.shape == (8, 32)

    def test_autoencoder_reconstruction_difference(self):
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(8, 30)
        reconstructed, _ = ae.forward(x)
        mse = functional.mse_loss(x, reconstructed)
        assert mse > 0

    def test_autoencoder_embedding_l2_normalizable(self):
        """Raw encoder output can be L2-normalized (as model/embed.py does)."""
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(16, 30)
        _, embedding = ae.forward(x)
        normed = functional.normalize(embedding, p=2, dim=1)
        norms = torch.norm(normed, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), atol=1e-6, rtol=1e-6)

    def test_autoencoder_gradient_flow(self):
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
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(10, 30)
        _, embedding = ae.forward(x)
        assert torch.isfinite(embedding).all()

    def test_embedding_no_nans(self):
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(10, 30)
        _, embedding = ae.forward(x)
        assert not torch.isnan(embedding).any()

    def test_embedding_cosine_similarity(self):
        ae = PlayerAutoencoder(input_dim=30, embedding_dim=32)
        ae.eval()
        x = torch.randn(2, 30)
        _, z1 = ae.forward(x)
        _, z2 = ae.forward(x)
        cosine_sim = functional.cosine_similarity(z1, z2, dim=1)
        assert (cosine_sim > 0.99).all()

    def test_embedding_dimension_mismatch(self):
        for dim in [16, 32, 64]:
            ae = PlayerAutoencoder(input_dim=30, embedding_dim=dim)
            ae.eval()
            x = torch.randn(8, 30)
            _, embedding = ae.forward(x)
            assert embedding.shape[1] == dim
