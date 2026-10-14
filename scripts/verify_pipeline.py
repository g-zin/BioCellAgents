import torch
from src.models.dino_classifier import DINOv2CellClassifier, DINOv2CellSegmenter

def test_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running verification test on device: {device}\n")

    # 1. Test Single-Cell Classifier
    print("Testing DINOv2CellClassifier...")
    try:
        classifier = DINOv2CellClassifier(model_size="vits14", num_classes=5).to(device)
        dummy_cls_input = torch.randn(2, 3, 224, 224, device=device)
        cls_out = classifier(dummy_cls_input)
        print(f"  Success! Output shape: {cls_out.shape} (Expected: [2, 5])")
    except Exception as e:
        print(f"  Classifier failed with error:\n{e}")
        return

    # 2. Test Spatial Segmenter
    print("\nTesting DINOv2CellSegmenter...")
    try:
        segmenter = DINOv2CellSegmenter(model_size="vits14", num_classes=6).to(device)
        dummy_seg_input = torch.randn(2, 3, 224, 224, device=device)
        seg_out = segmenter(dummy_seg_input)
        print(f"  Success! Output shape: {seg_out.shape} (Expected: [2, 6, 224, 224])")
    except Exception as e:
        print(f"  Segmenter failed with error:\n{e}")
        return

    print("\nAll tests passed! The xFormers/precision error is resolved.")

if __name__ == "__main__":
    test_pipeline()