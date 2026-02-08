import os
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model():
    return smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=4,
        activation=None
    )

def find_checkpoint_file(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):

        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith((".pth", ".pt")):
                    return os.path.join(root, f)
    
        for candidate in ("data", "checkpoints", "best"):
            candidate_path = os.path.join(path, candidate)
            if os.path.isdir(candidate_path):
                found = find_checkpoint_file(candidate_path)
                if found:
                    return found
    raise FileNotFoundError(f"No .pth/.pt file found at or under: {path}")

def strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("module.") for k in keys):
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return new_state
    return state_dict

def load_model_auto(checkpoint_path, model_builder, device):
    ckpt_file = find_checkpoint_file(checkpoint_path)
    print(f"Loading checkpoint file: {ckpt_file}")
    loaded = torch.load(ckpt_file, map_location=device)

    if isinstance(loaded, torch.nn.Module):
        model = loaded
        model.to(device)
        model.eval()
        print("Loaded full model object from checkpoint.")
        return model

 
    if isinstance(loaded, dict):
        if "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        elif "state_dict" in loaded:
            state_dict = loaded["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in loaded.values()):
            state_dict = loaded
        else:

            possible = None
            for v in loaded.values():
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    possible = v
                    break
            if possible is not None:
                state_dict = possible
            else:
                raise RuntimeError("Loaded checkpoint dict doesn't contain a recognizable state_dict key.")

        state_dict = strip_module_prefix(state_dict)
        model = model_builder()

        try:
            model.load_state_dict(state_dict, strict=True)
            print("Loaded state_dict into model (strict=True).")
        except Exception as e:
            print("Strict load failed:", e)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded state_dict into model (strict=False). Some keys may have been ignored/mismatched.")
        model.to(device)
        model.eval()
        return model

    raise RuntimeError("Unknown checkpoint format (not module nor dict).")

checkpoint_path = "/home/javra/work/yolo/object_detection_frame/Best_checkpoint_train3.pth"  
model = load_model_auto(checkpoint_path, make_model, DEVICE)

def predict_mask(model, image, device, input_size=(256, 256)):
    orig_size = image.shape[:2]  # (h, w)
    img_resized = cv2.resize(image, input_size)
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)

        if isinstance(pred, dict):
            for v in pred.values():
                if isinstance(v, torch.Tensor):
                    pred = v
                    break

        if isinstance(pred, torch.Tensor) and pred.dim() == 4 and pred.shape[1] > 1:
            pred = torch.argmax(pred, dim=1)

        pred_np = pred.squeeze(0).cpu().numpy()

    pred_resized = cv2.resize(pred_np.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    return pred_resized

def create_overlay(image, mask, alpha=0.5):
    class_colours = {
        0: (0, 0, 0),
        1: (128, 0, 128),
        2: (255, 0, 0),
        3: (0, 255, 0),
    }
    overlay = np.zeros_like(image, dtype=np.uint8)
    for class_id, colour in class_colours.items():
        if class_id == 0:
            continue
        class_mask = (mask == class_id)
        for c in range(3):
            overlay[..., c][class_mask] = colour[c]

    blended_image = image.copy()
    class_mask = (mask != 0)
    for c in range(3):
        blended_image[..., c][class_mask] = (
            image[..., c][class_mask] * (1 - alpha) +
            overlay[..., c][class_mask] * alpha
        ).astype(np.uint8)
    return blended_image

image_path = "nsd/test/109_jpg.rf.00a982a54586eefbd1b0a572f1e0c25d.jpg"
image = np.array(Image.open(image_path).convert("RGB"))

pred_mask = predict_mask(model, image, DEVICE)
blended = create_overlay(image, pred_mask, alpha=0.5)

plt.imshow(blended)
plt.axis("off")
plt.savefig("pred_overlay_from_checkpoint.png", bbox_inches="tight", pad_inches=0)
plt.close()
print("Overlay saved as pred_overlay_from_checkpoint.png")
