from torch.utils.data import Dataset
from torchvision import transforms
from .utils import get_path_label_pairs
from PIL import Image
import torch

import os

class BrainHemorrhageDataset(Dataset):
    """Dataset of two types of Brain Hemorrhage:
        1. Subdurral Haematoma:
        2. Subarachnial Haemorrhage
   
      Arguments ---------------
      
         - filepath (str): directory to textfile listing image datasets path + labels value.
          Textfile must contains information about img path and label values for each row.
          Img path and label values should be separated by comma by default (.txt),
          unless it follows a specific file extension separator (e.g tsv)
          
        - size (sequence or int): size of resized img dataset. If sequence is given, 
            it specify the shape of each img, else img is assumed to be resized as square img.
        
        - mean (float): mean for img norm

        - std (float): standard deviation for img norm
      
    """

    def __init__(self, filepath, size, mean=0.5, std=0.5):
        """Construct BrainHemorrhage Dataset Object
        """
        self.size = size
        self.mean = mean
        self.std = std
        
        # check if filepath exists and contains correct path to images.
        if os.path.exist(filepath):
            self.filepath = filepath
            self.img_paths, self.labels = get_path_labels_pairs(self.filepath)
        else:
            raise FileNotFoundError("Path to image-label file: {} cannot be found")
        
        self.transforms = transforms.Compose([transforms.Resize(size),
                                              transforms.CenterCrop(size),
                                              transforms.GrayScale(3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std),
                                             ])

    def __getitem__(self, index):
        """Get one image dataset
        """
        img = Image.open(self.img_paths[index])
        img = self.transforms(img)
        
        label = self.labels[index]
        
        
        