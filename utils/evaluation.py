import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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