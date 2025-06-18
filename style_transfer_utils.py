# style_transfer_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Convert tensor back to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

# Content and style loss layers
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Run style transfer
def run_style_transfer(vgg, content, style, content_layers, style_layers, num_steps=300, style_weight=1e6, content_weight=1):
    model = vgg.features.to(device).eval()

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad_(False)

    content_features = {}
    style_features = {}
    x = content.clone().requires_grad_(True)

    for name, layer in model._modules.items():
        content = layer(content)
        style = layer(style)
        if name in content_layers:
            content_features[name] = content
        if name in style_layers:
            style_features[name] = gram_matrix(style)

    optimizer = torch.optim.Adam([x], lr=0.003)

    for step in range(1, num_steps+1):
        target_features = {}
        x_features = x
        for name, layer in model._modules.items():
            x_features = layer(x_features)
            if name in content_layers:
                target_features[name] = x_features
            if name in style_layers:
                target_features[name + '_gram'] = gram_matrix(x_features)

        content_loss = 0
        style_loss = 0

        for name in content_layers:
            content_loss += F.mse_loss(target_features[name], content_features[name])

        for name in style_layers:
            style_loss += F.mse_loss(target_features[name + '_gram'], style_features[name])

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return x

