import argparse
import torch
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Define command-line arguments
parser = argparse.ArgumentParser(description='Predict flower name from an image.')
parser.add_argument('input', type=str, help='Path to the input image')
parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

# Parse the command-line arguments
args = parser.parse_args()

# Check if GPU is available and whether to use it
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

# Load the checkpoint
checkpoint = torch.load(args.checkpoint)

# Load the mapping of categories to real names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the model architecture
if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)  # Use the same architecture used during training
else:
    raise ValueError("Unsupported model architecture")

# Load the model state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Define data transformation for the input image
preprocess = transforms.Compose
