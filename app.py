import streamlit as st
import torch
from torch.autograd import Variable
from collections import namedtuple
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import sys
import random
from PIL import Image
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import DatasetFolder
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mean and standard deviation used for training
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def test_transform(image_size=None):
    """ Transforms for test image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


#Components of Transformer Net """
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x



def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

def inference(image_path,checkpoint_model):
    os.makedirs("results", exist_ok=True)
    img_name = os.path.basename(image_path).split('.')[0]
    model_name = os.path.basename(checkpoint_model).split('.')[0]

    transform = test_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model))
    transformer.eval()

    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()
    
    # Save image
    output_path = f"results/{img_name}-{model_name}.jpg"
    save_image(stylized_image, output_path)
    print(f"Image Saved at {output_path}")
    return output_path


IMG_PATH = 'trees.jpeg'
MODEL_PATH = 'models/anime_art.pth'
st.sidebar.image('dashtoon_logo.png')
style = st.sidebar.radio(
    "Choose the style of your liking",
    [":rainbow[Modern Art]", "Game Art", "Manga Sketch"],
    # captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."]
    )

if style == ':rainbow[Modern Art]':
    style_image_path = 'styles/modern_art.jpg'
    MODEL_PATH = 'models/modern_art.pth'
elif style == "Game Art":
    style_image_path = 'styles/gta.jpg'
    MODEL_PATH = 'models/game.pth'
elif style == 'Manga Sketch':
    style_image_path = 'styles/kakshi.jpg'
    MODEL_PATH = 'models/manga_sketch.pth'

st.header("Dashtoon Image Styler")
st.sidebar.title("Dashtoon Image Styler")
uploaded_file = st.sidebar.file_uploader("Upload the Image", type = ['jpeg', 'jpg', 'png'])
button = st.sidebar.button("upload")

col1, col2, col3 = st.columns(3)  
with col1:
    st.image(Image.open('styles/modern_art.jpg').resize((300,200), Image.BILINEAR) , caption= "Modern Art")
with col2:
    st.image(Image.open('styles/gta.jpg').resize((300,200), Image.BILINEAR) ,caption= "Game Art")
with col3:
    st.image(Image.open('styles/kakshi.jpg').resize((300,200), Image.BILINEAR), caption= "Manga Sketch")

if button:
    st.header("Results")
    original_image = Image.open(uploaded_file)
    style_image = Image.open(style_image_path)
    IMG_PATH = os.path.join('original_images', uploaded_file.name)
    original_image.save(IMG_PATH)
    # plt.imsave(IMG_PATH, original_image)
    # cv2.imwrite(IMG_PATH, original_image)
    modified_image_path = inference(IMG_PATH, MODEL_PATH)
    modified_image = Image.open(modified_image_path)

    col1, col2, col3 = st.columns(3)  
    with col1:
        st.image(original_image, caption= "Original Image", use_column_width=True)
    with col2:
        st.image(style_image, caption= "style image", use_column_width=True)
    with col3:
        st.image(modified_image, caption= "Output image", use_column_width=True)

