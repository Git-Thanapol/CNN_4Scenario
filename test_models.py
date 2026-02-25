import torch
from src.models import ResNet50, MobileNetV3, EfficientNetV2, DenseNet, AlexNet

def test_models():
    models = {
        "ResNet50": ResNet50(n_classes=4),
        "MobileNetV3": MobileNetV3(n_classes=4),
        "EfficientNetV2": EfficientNetV2(n_classes=4),
        "DenseNet": DenseNet(n_classes=4),
        "AlexNet": AlexNet(n_classes=4)
    }

    # Dummy spectrogram input: Batch=2, Channels=1, Freq=128, Time=1024
    dummy_input = torch.randn(2, 1, 128, 1024)

    print("Testing models with dummy input (2, 1, 128, 1024)...")
    for name, model in models.items():
        try:
            model.eval() # Set to eval to avoid BatchNorm issues with small batch
            with torch.no_grad():
                out, features = model(dummy_input)
            print(f"[OK] {name}: Output shape {out.shape}, Features shape {features.shape}")
        except Exception as e:
            print(f"[ERROR] {name}: {str(e)}")

if __name__ == "__main__":
    test_models()
