import torch
import torch.nn as nn
import pytest
from src.shortcut_models_pytorch.model import TimestepEmbedder, LabelEmbedder, PatchEmbed, MlpBlock, DiT, DiTBlock, FinalLayer

@pytest.fixture
def embedder():
    hidden_size = 64
    freq_size = 256
    return TimestepEmbedder(hidden_size=hidden_size, frequency_embedding_size=freq_size)

def test_timestep_embedder_init(embedder):
    assert embedder.hidden_size == 64
    assert embedder.frequency_embedding_size == 256
    assert len(embedder.net) == 3
    assert isinstance(embedder.net[0], nn.Linear)
    assert isinstance(embedder.net[1], nn.SiLU)
    assert isinstance(embedder.net[2], nn.Linear)

def test_timestep_embedding(embedder):
    # Test single timestep
    t = torch.tensor([0.5])
    embedding = embedder.timestep_embedding(t)
    assert embedding.shape == (1, embedder.frequency_embedding_size)
    
    # Test batch of timesteps
    batch_t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    batch_embedding = embedder.timestep_embedding(batch_t)
    assert batch_embedding.shape == (5, embedder.frequency_embedding_size)
    
    # Test output range
    assert torch.all(batch_embedding >= -1.0)
    assert torch.all(batch_embedding <= 1.0)
    
    # Test periodic nature
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    emb1 = embedder.timestep_embedding(t1)
    emb2 = embedder.timestep_embedding(t2)
    assert not torch.allclose(emb1, emb2)

def test_forward(embedder):
    # Test single input
    t = torch.tensor([0.5])
    output = embedder.forward(t)
    assert output.shape == (1, embedder.hidden_size)
    
    # Test batch input
    batch_t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    batch_output = embedder.forward(batch_t)
    assert batch_output.shape == (5, embedder.hidden_size)
    
    # Test different inputs produce different outputs
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    out1 = embedder.forward(t1)
    out2 = embedder.forward(t2)
    assert not torch.allclose(out1, out2)

def test_invalid_inputs(embedder):
    # Test input validation
    with pytest.raises(RuntimeError):
        embedder.forward(torch.tensor([[0.5, 0.6]]))  # Wrong shape
        
    with pytest.raises(RuntimeError):
        embedder.timestep_embedding(torch.tensor([[0.5, 0.6]]))  # Wrong shape

def test_numerical_stability(embedder):
    # Test extreme values
    t_extreme = torch.tensor([1e-6, 1e6])
    output = embedder.forward(t_extreme)
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))


@pytest.fixture
def label_embedder():
    num_classes = 10
    hidden_size = 64
    return LabelEmbedder(num_classes=num_classes, hidden_size=hidden_size)

def test_label_embedder_init(label_embedder):
    assert label_embedder.num_classes == 10
    assert label_embedder.hidden_size == 64
    assert label_embedder.embedding_table.num_embeddings == 11  # num_classes + 1
    assert label_embedder.embedding_table.embedding_dim == 64
    
def test_label_embedder_forward(label_embedder):
    # Test single label
    label = torch.tensor([5])
    embedding = label_embedder.forward(label)
    assert embedding.shape == (1, label_embedder.hidden_size)
    
    # Test batch of labels
    batch_labels = torch.tensor([0, 3, 5, 7, 9])
    batch_embedding = label_embedder.forward(batch_labels)
    assert batch_embedding.shape == (5, label_embedder.hidden_size)

def test_embedding_properties(label_embedder):
    # Test embedding table initialization
    embedding_weights = label_embedder.embedding_table.weight
    assert torch.abs(embedding_weights.mean()) < 0.1  # Roughly centered around 0
    assert 0.01 < embedding_weights.std() < 0.03  # Close to initialization std of 0.02
    
    # Test different labels produce different embeddings
    label1 = torch.tensor([0])
    label2 = torch.tensor([1])
    emb1 = label_embedder.forward(label1)
    emb2 = label_embedder.forward(label2)
    assert not torch.allclose(emb1, emb2)

def test_label_embedder_invalid_inputs(label_embedder):
    # Test out of range labels
    with pytest.raises(IndexError):
        invalid_label = torch.tensor([11])  # Above num_classes + 1
        label_embedder.forward(invalid_label)
    
    with pytest.raises(IndexError):
        negative_label = torch.tensor([-1])
        label_embedder.forward(negative_label)


@pytest.fixture
def patch_embed():
    patch_size = 4
    hidden_size = 64
    return PatchEmbed(patch_size=patch_size, hidden_size=hidden_size)

def test_patch_embed_init(patch_embed):
    assert patch_embed.patch_size == 4
    assert patch_embed.hidden_size == 64
    assert patch_embed.conv.kernel_size == (4, 4)
    assert patch_embed.conv.stride == (4, 4)
    assert patch_embed.conv.bias is not None

def test_patch_embed_forward_single_image(patch_embed):
    x = torch.randn(1, 3, 16, 16)  # Single image with 3 channels, 16x16 pixels
    output = patch_embed.forward(x)
    assert output.shape == (1, 16, 64)  # (batch_size, num_patches, hidden_size)

def test_patch_embed_forward_batch_images(patch_embed):
    x = torch.randn(8, 3, 16, 16)  # Batch of 8 images with 3 channels, 16x16 pixels
    output = patch_embed.forward(x)
    assert output.shape == (8, 16, 64)  # (batch_size, num_patches, hidden_size)

@pytest.fixture
def mlp_block():
    input_dim = 128
    mlp_dim = 256
    out_dim = 128
    dropout_rate = 0.1
    return MlpBlock(input_dim=input_dim, mlp_dim=mlp_dim, out_dim=out_dim, dropout_rate=dropout_rate)

def test_mlp_block_init(mlp_block):
    assert mlp_block.input_dim == 128
    assert mlp_block.mlp_dim == 256
    assert mlp_block.out_dim == 128
    assert mlp_block.dropout_rate == 0.1
    assert isinstance(mlp_block.fc[0], nn.Linear)
    assert isinstance(mlp_block.fc[1], nn.GELU)
    assert isinstance(mlp_block.fc[2], nn.Dropout)
    assert isinstance(mlp_block.fc[3], nn.Linear)
    assert isinstance(mlp_block.fc[4], nn.Dropout)

def test_mlp_block_forward(mlp_block):
    x = torch.randn(32, 10, 128)  # Batch of 32, sequence length 10, embedding size 128
    output = mlp_block.forward(x)
    assert output.shape == (32, 10, 128)  # (batch_size, sequence_length, out_dim)

def test_mlp_block_dropout(mlp_block):
    x = torch.randn(32, 10, 128)  # Batch of 32, sequence length 10, embedding size 128
    mlp_block.eval()  # Disable dropout
    output_no_dropout = mlp_block.forward(x)
    mlp_block.train()  # Enable dropout
    output_with_dropout = mlp_block.forward(x)
    assert not torch.allclose(output_no_dropout, output_with_dropout)



@pytest.fixture
def dit_block():
    hidden_size = 64
    num_heads = 4
    return DiTBlock(hidden_size=hidden_size, num_heads=num_heads)

def test_dit_block_init(dit_block):
    assert dit_block.hidden_size == 64
    assert dit_block.num_heads == 4
    assert dit_block.mlp_ratio == 4.0
    assert isinstance(dit_block.c_map, nn.Sequential)
    assert isinstance(dit_block.layer_norm, nn.LayerNorm)
    assert isinstance(dit_block.layer_norm2, nn.LayerNorm)

def test_dit_block_forward():
    block = DiTBlock(hidden_size=64, num_heads=4)
    batch_size = 4
    seq_len = 16
    
    x = torch.randn(batch_size, seq_len, 64)
    c = torch.randn(batch_size, 64)
    
    output = block.forward(x, c)
    assert output.shape == (batch_size, seq_len, 64)

@pytest.fixture
def final_layer():
    return FinalLayer(patch_size=4, out_channels=3, hidden_size=64)

def test_final_layer_init(final_layer):
    assert final_layer.patch_size == 4
    assert final_layer.out_channels == 3
    assert final_layer.hidden_size == 64
    assert isinstance(final_layer.c_map, nn.Sequential)
    assert isinstance(final_layer.layer_norm, nn.LayerNorm)
    assert isinstance(final_layer.linear, nn.Linear)

def test_final_layer_forward(final_layer):
    batch_size = 4
    seq_len = 16
    
    x = torch.randn(batch_size, seq_len, 64)
    c = torch.randn(batch_size, 64)
    
    output = final_layer.forward(x, c)
    expected_size = final_layer.patch_size * final_layer.patch_size * final_layer.out_channels
    assert output.shape == (batch_size, seq_len, expected_size)

@pytest.fixture
def dit_model():
    return DiT(
        patch_size=4,
        hidden_size=64,
        conditioning_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        out_channels=3,
        num_classes=10
    )

def test_dit_init(dit_model):
    assert dit_model.patch_size == 4
    assert dit_model.hidden_size == 64
    assert dit_model.depth == 2
    assert dit_model.num_heads == 4
    assert len(dit_model.dit_blocks) == 2
    assert isinstance(dit_model.patch_embed, PatchEmbed)
    assert isinstance(dit_model.time_embedder, TimestepEmbedder)
    assert isinstance(dit_model.label_embedder, LabelEmbedder)
    assert isinstance(dit_model.final_layer, FinalLayer)

def test_dit_forward(dit_model):
    batch_size = 4
    image_size = 16
    channels = 3
    
    x = torch.randn(batch_size, channels, image_size, image_size)
    t = torch.rand(batch_size)
    dt = torch.rand(batch_size)
    y = torch.randint(0, 10, (batch_size,))
    
    output, logvars, activations = dit_model.forward(x, t, dt, y, return_activations=True)
    
    # Check output shapes
    assert output.shape == (batch_size, channels, image_size, image_size)
    assert logvars.shape == (batch_size, 1)
    
    # Check activations
    assert 'patch_embed' in activations
    assert 'conditioning' in activations
    assert 'dit_block_0' in activations
    assert 'dit_block_1' in activations
    assert 'final_layer' in activations

def test_dit_ignore_dt(dit_model):
    batch_size = 4
    image_size = 16
    channels = 3
    
    dit_model.ignore_dt = True
    x = torch.randn(batch_size, channels, image_size, image_size)
    t = torch.rand(batch_size)
    dt = torch.rand(batch_size)
    y = torch.randint(0, 10, (batch_size,))
    
    output1 = dit_model.forward(x, t, dt, y)
    output2 = dit_model.forward(x, t, torch.zeros_like(dt), y)
    
    assert torch.allclose(output1, output2)