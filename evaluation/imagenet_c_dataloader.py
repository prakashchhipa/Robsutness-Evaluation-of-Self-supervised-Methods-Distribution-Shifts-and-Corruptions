import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch

from PIL import Image
corruptions = { 'blur' : {'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'},
                   'digital': {'contrast',  'elastic_transform', 'jpeg_compression', 'pixelate'},
                   'extra': {'gaussian_blur', 'saturate', 'spatter', 'speckle_noise'},
                   'noise': {'gaussian_noise', 'impulse_noise', 'shot_noise'}, 
                   'weather' :{'brightness', 'fog', 'frost', 'snow'}}

curruptions_subtypes = {'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
                        'contrast',  'elastic_transform', 'jpeg_compression', 'pixelate',
                        'gaussian_blur', 'saturate', 'spatter', 'speckle_noise',
                        'gaussian_noise', 'impulse_noise', 'shot_noise',
                        'brightness', 'fog', 'frost', 'snow'
                        }

def get_imagenet_baseset(type: str, sub_type: str, root_path: str, transform: callable, difficulty: int = 1) -> Dataset:
    assert (difficulty > 0)
    assert (difficulty <= 5)
    assert (type in corruptions)
    assert (sub_type in corruptions[type]) 
    path = f"{root_path}/{type}/{sub_type}/{difficulty}"
    return torchvision.datasets.ImageFolder(path, transform)
    



class ImagenetCDataloader(object):
    def __init__(self, path: str, transform: transforms, **kwargs) -> None:
        self.datasets: Dict[str, DataLoader] = {}
        for type, subtype in corruptions.items():
            for sub in subtype:
                for diff in range(1,6):
                    self.datasets[f"{type}_{sub}_{diff}"] = DataLoader( get_imagenet_baseset(type,sub, path, transform, diff),
                        **kwargs
                        )
                    
    def get_from_type(self, type: str ) -> List[Tuple[str,DataLoader]]:
        return [(k, x) for k,x in self.datasets.items() if k.startswith(type)]
    
    def get_from_sub_type(self, type:str, sub_type: str ) -> List[Tuple[str,DataLoader]]:
        return [(k, x) for k,x in self.datasets.items() if k.startswith(type+'_'+sub_type)]
    
    def get_from_diff(self, type:str, sub_type: str, diff: int ) -> List[Tuple[str,DataLoader]]:
        return [(k, x) for k,x in self.datasets.items() if k == f"{type}_{sub_type}_{diff}"]