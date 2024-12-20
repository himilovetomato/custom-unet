import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path




@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


def evaluate_and_visualize(model, dataloader, device, epoch, save_dir='eval_results'):
    """Evaluate model performance and save visualization results"""
    model.eval()
    save_dir = Path(save_dir) / f'epoch_{epoch}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get images and masks
            images = batch['image'].to(device)
            true_masks = batch['mask'].to(device)
            
            # Generate predictions
            masks_pred = model(images)
            
            # Convert predictions
            if model.n_classes == 1:
                probs = torch.sigmoid(masks_pred)
                masks_pred = (probs > 0.5).float()
            else:
                probs = torch.softmax(masks_pred, dim=1)
                masks_pred = probs.argmax(dim=1)

            # Save visualization for first image in batch
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            if img.shape[2] == 1:  # Grayscale
                axes[0].imshow(img.squeeze(), cmap='gray')
            else:  # RGB
                axes[0].imshow(img)
            axes[0].set_title('Input Image')
            
            # True mask
            axes[1].imshow(true_masks[0].cpu(), cmap='tab20')
            axes[1].set_title('True Mask')
            
            # Predicted mask
            axes[2].imshow(masks_pred[0].cpu(), cmap='tab20')
            axes[2].set_title('Predicted Mask')
            
            # Save plot
            plt.savefig(save_dir / f'sample_{batch_idx}.png')
            plt.close()
            
            # Only save first few samples
            if batch_idx >= 4:
                break

    model.train()
