import pytest


def test_model_forward_backward_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from voice_localization.model import build_model

    model = build_model()
    inputs = torch.randn(2, 2, 61, 81)
    target = torch.rand(2, 1, 61, 81)
    output = model(inputs)
    assert output.shape == target.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_wave_unet_keeps_multichannel_waveform_shape_and_gradients():
    torch = pytest.importorskip("torch")
    from voice_localization.enhancement import build_wave_unet

    model = build_wave_unet(input_channels=32, output_channels=8, base_channels=8, depth=4)
    inputs = torch.randn(1, 32, 2048)
    output = model(inputs)
    assert output.shape == (1, 8, 2048)
    assert torch.isfinite(output).all()
    output.abs().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

