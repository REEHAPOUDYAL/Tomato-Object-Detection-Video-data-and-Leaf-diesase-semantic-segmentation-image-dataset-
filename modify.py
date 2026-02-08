import torch

checkpoint_path = "runs/detect/train732/weights/last.pt"

checkpoint = torch.load(checkpoint_path, weights_only=False)

print(f"Current recorded epoch: {checkpoint['epoch']}")
checkpoint['epoch'] = 48

torch.save(checkpoint, checkpoint_path)
print(f"Epoch count updated to: {checkpoint['epoch']}")
print("Checkpoint file modified. You can now try resuming training.")