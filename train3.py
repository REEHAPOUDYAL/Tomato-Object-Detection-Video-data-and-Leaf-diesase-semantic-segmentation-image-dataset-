import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, UnidentifiedImageError
import matplotlib
matplotlib.use('Agg')
writer = SummaryWriter(log_dir="runs/segmentation_logs")



mask = cv2.imread("/home/javra/work/yolo/object_detection_frame/nsd/train/1_jpg.rf.16b1d846b87c3ba04b73e253d7dded0a_mask.png", cv2.IMREAD_UNCHANGED)
print(np.unique(mask))

class LeafDiseaseDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.split_dir) if f.endswith(".jpg")]
        self.class_index = {0: "background", 1: "early_blight", 2: "late_blight", 3: "leaf_miner"}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.split_dir, img_filename)
        image = np.array(Image.open(img_path).convert("RGB"))

        base_name = os.path.splitext(img_filename)[0]
        mask_filename = base_name + "_mask.png"
        mask_path = os.path.join(self.split_dir, mask_filename)
        mask = np.array(Image.open(mask_path).convert("L"))

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.int64))
        return image, mask


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataloaders(root_dir, batch_size=8):
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], additional_targets={'mask': 'mask'})

    val_test_transform = A.Compose([
        A.Resize(256, 256),
    ], additional_targets={'mask': 'mask'})

    datasets = {
        "train": LeafDiseaseDataset(root_dir, split="train", transform=train_transform),
        "valid": LeafDiseaseDataset(root_dir, split="valid", transform=val_test_transform),
        "test": LeafDiseaseDataset(root_dir, split="test", transform=val_test_transform),
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn
        )
        for split in ["train", "valid", "test"]
    }

    return dataloaders


class ComboLoss(nn.Module):
    def __init__(self, mode='multiclass', dice_weight=1.0, focal_weight=1.0, alpha=None):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=mode, from_logits=True)
        self.focal = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=2.0)
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return self.dw * dice_loss + self.fw * focal_loss


def multiclass_iou(preds, labels, num_classes=4):
    smooth = 1e-6
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        ious.append((intersection + smooth) / (union + smooth))
    return np.mean(ious)


def multiclass_fscore(preds, labels, num_classes=4):
    preds = torch.argmax(preds, dim=1)
    f1_per_class = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        tp = (pred_inds & target_inds).sum().item()
        fp = (pred_inds & ~target_inds).sum().item()
        fn = (~pred_inds & target_inds).sum().item()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f1_per_class.append(f1)
    return np.mean(f1_per_class)


def multiclass_pixel_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.numel()

def multiclass_dice_score(preds, labels, num_classes=4):
    smooth = 1e-6
    preds = torch.argmax(preds, dim=1)
    dices = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        dices.append((2 * intersection + smooth) / (union + smooth))
    return np.mean(dices)



if __name__ == "__main__":
    root_dir = "/home/javra/work/yolo/object_detection_frame/nsd"
    dataloaders = get_dataloaders(root_dir, batch_size=4)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
        activation=None
    )
    for param in model.encoder.parameters():
        param.requires_grad = False


    for param in model.encoder.features[-1].parameters():
        param.requires_grad = True
    
    model.to(DEVICE)
    loss_function = ComboLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    checkpoint_path = "Best_checkpoint_train3.pth" 

    start_epoch = 0
    max_epochs = 64
    best_loss = float('inf')
    patience = 15
    counter = 0

    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        counter = checkpoint.get('counter', 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1} Training"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss_value = loss_function(outputs, masks)
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()

        train_loss /= len(dataloaders['train'])

        model.eval()
        val_loss = 0
        val_ious, val_f1s, val_accs, val_dices = [], [], [], []

        with torch.no_grad():
            for images, masks in tqdm(dataloaders['valid'], desc=f"Epoch {epoch+1} Validation"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss_value = loss_function(outputs, masks)
                val_loss += loss_value.item()
                val_ious.append(multiclass_iou(outputs, masks))
                val_f1s.append(multiclass_fscore(outputs, masks))
                val_accs.append(multiclass_pixel_accuracy(outputs, masks))
                val_dices.append(multiclass_dice_score(outputs , masks))

        val_loss /= len(dataloaders['valid'])
        iou = np.mean(val_ious)
        f1 = np.mean(val_f1s)
        acc = np.mean(val_accs)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, IoU {iou:.4f}, F1 {f1:.4f}, Acc {acc:.4f}")
        writer.add_scalar("IoU/Val", iou, epoch)
        writer.add_scalar("Fscore/Val", f1, epoch)
        writer.add_scalar("Accuracy/Val", acc, epoch)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss,
                'counter': 0,
            }, checkpoint_path)
            torch.save(model.state_dict(), "best.pth")
            print(f"Checkpoint and best model saved at epoch {epoch}")
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break
            
if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded successfully. Best loss from training: {checkpoint['best_loss']:.4f}")
else:
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'.")
        exit()

   
dataloaders = get_dataloaders(root_dir, batch_size=4)
loss_function = ComboLoss(mode='multiclass')

print("\nEvaluating on Validation Set:")
val_loss, val_iou, val_f1, val_acc, val_dice = evaluate_model(model, dataloaders['valid'], DEVICE, loss_function)
print(f"Validation Metrics: Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice Score: {val_dice:.4f}, F1 Score: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
print("---------------------------------------------------------------------------------------------------------------------")

print("\nEvaluating on Test Set:")
test_loss, test_iou, test_f1, test_acc, test_dice = evaluate_model(model, dataloaders['test'], DEVICE, loss_function)
print(f"Test Metrics: Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice Score: {test_dice:.4f}, F1 Score: {test_f1:.4f}, Accuracy: {test_acc:.4f}")
writer.close()

                
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

image = np.array(Image.open("nsd/test/1_jpg.rf.39b8689b2b82c80a4ed93028499b725e.jpg").convert("RGB"))
mask = np.array(Image.open("nsd/test/1_jpg.rf.39b8689b2b82c80a4ed93028499b725e.png"))
mask = mask.astype(np.uint8)
blended = create_overlay(image, mask, alpha=0.5)

plt.imshow(blended)
plt.axis("off")
plt.savefig("overlay.png")
plt.close()



