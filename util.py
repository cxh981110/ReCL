from methods import mixup_data
from torchvision.transforms.functional import to_pil_image
import torch
class ThreeCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        
        # x = to_pil_image(x)
        # # print("adad",type(self.transform(x)))
        return [self.transform(x), self.transform(x)]



