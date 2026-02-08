import torch
import segmentation_models_pytorch as smp

checkpoint_path = "Best_checkpoint_train3.pth"
onnx_path = "best.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4,
    activation=None
)

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(DEVICE)
model.eval()
dummy_input = torch.randn(1, 3, 256, 256, device=DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'}
    }
)

print(f"Segmentation model exported to {onnx_path}")
