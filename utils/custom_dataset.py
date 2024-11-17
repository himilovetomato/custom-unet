from .data_loading import BasicDataset
from pathlib import Path

class CustomDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1.0):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')
        
    @classmethod
    def from_fold(cls, fold_path: str, split='train', scale=1.0):
        """Create dataset from fold directory structure"""
        fold_path = Path(fold_path)
        images_dir = fold_path / split / 'images'
        masks_dir = fold_path / split / 'masks'
        return cls(images_dir, masks_dir, scale)