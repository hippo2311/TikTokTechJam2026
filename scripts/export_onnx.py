"""Export the Reelistic DINOv3 detector to ONNX.

The exported graph accepts ImageNet-normalized NCHW RGB tensors and returns a
single logit. Apply sigmoid to the output to get the AI-generated probability.
"""
import argparse
import hashlib
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "models" / "reelistic_dino" / "src"))
    from robust_aigc.models import DINOv3Forensic

    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)
    model = DINOv3Forensic(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model_state")), strict=True)
    model.eval()

    class LogitOnly(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, image):
            return self.detector(image)["logits"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        LogitOnly(model),
        torch.randn(1, 3, config["data"].get("image_size", 224), config["data"].get("image_size", 224)),
        str(output),
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    import onnx
    onnx.checker.check_model(onnx.load(str(output), load_external_data=True))
    print(f"Exported {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("ONNX checker: OK")
    print(f"SHA-256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
