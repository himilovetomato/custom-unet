import argparse
from pathlib import Path
from utils.custom_dataset import CustomDataset
from train import train_model
from unet import UNet
import torch
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='Fold number (1-5)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--data-dir', type=str, default='cross_validation',
                       help='Path to dataset root directory')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets for current fold using custom path
    fold_path = Path(args.data_dir) / f'fold_{args.fold}'
    train_set = CustomDataset.from_fold(fold_path, 'train', args.scale)
    val_set = CustomDataset.from_fold(fold_path, 'val', args.scale)
    
    # Get number of classes from dataset
    n_classes = len(train_set.mask_values)
    logging.info(f'Number of classes in dataset: {n_classes}')
    
    # Initialize model with correct number of classes
    model = UNet(n_channels=3, n_classes=n_classes)  # Updated to use dataset classes
    model = model.to(device)
    
    # Train
    try:
        train_model(
            model=model,
            device=device,
            train_set=train_set,
            val_set=val_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            img_scale=args.scale,
            amp=args.amp
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')

if __name__ == '__main__':
    main()