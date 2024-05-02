import torch
import torch.nn as nn
from torch.nn import functional as F
from model.ResUNet import ResUNet
from model.TransUNet import TransUNet
from model.UNet import UNet
from utils.data_loader import create_test_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import shutil
import numpy as np
from PIL import Image
from skimage import color, io


def dataload(file_path,batch_size,n_w):
    test_loader = create_test_loader(file_path, batch_size=batch_size, test_size=0.2, 
                                                random_seed=42, n_w=n_w)
    return test_loader

def setup(in_channels,out_channels,n_layers=5,bn_layers=2,model_path=None,model_type=1):
    assert len(in_channels) == n_layers and len(out_channels) == n_layers, \
    'Error: channels should be same as number of layers'
    if model_type == 0:
        print("Using simple UNet")
        model = UNet(in_channels=in_channels,out_channels=out_channels,
                        blocks=n_layers,bn_blocks=bn_layers)
    elif model_type == 1:
        print("Using ResUNet")
        model = ResUNet(in_channels=in_channels,out_channels=out_channels,
                blocks=n_layers,bn_blocks=bn_layers)
    else:
        print("Using TransUNet")
        model = TransUNet(in_channels=in_channels,out_channels=out_channels,
                blocks=n_layers,bn_blocks=bn_layers)
                    
    if model_path is not None:
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("Couldn't load model weights")

    return model

def save_img(labels, outputs, idx):
    outputs = outputs.permute(0,2,3,1).cpu().numpy()
    labels = labels.permute(0,2,3,1).cpu().numpy()

    for i in range(len(outputs)):
        rgb_image = (outputs * 255).astype(np.uint8)
        image_path = os.path.join("predictions", f'image_orig_{idx}.png')
        io.imsave(image_path, rgb_image)
        
        rgb_image = (labels * 255).astype(np.uint8)
        image_path = os.path.join("predictions", f'image_pred_{idx}.png')
        io.imsave(image_path, rgb_image)

def predict(test_loader,model,device,num_batches=None):
    if os.path.exists("predictions"):
        shutil.rmtree("predictions")
    os.mkdir("predictions")

    print(f"Using device {device}")

    if num_batches is None:
        num_batches = len(test_loader)

    test_iterator = iter(test_loader)

    for i in  range(num_batches):
        images,labels = next(test_iterator)
            
        model.eval()
        # Move tensors to configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Calculate accuracy
        outputs = model(images)

        save_img(images, labels, outputs, i)