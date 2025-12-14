from typing import Any, Callable, Optional
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class MMImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            classes_exclude_ignored,
            transform=None,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.classes_exclude_ignored = classes_exclude_ignored
        
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        
        classes = self.classes_exclude_ignored
        # text = f"a photo of a {classes[label].replace('-', ' ').replace('_', ' ').lower()}" # should be ground truth
        text = f"a photo of a {classes[label]}" # should be ground truth
        
        x = {'images': image, 'texts': text, 'for_training': False}
        return x, label
