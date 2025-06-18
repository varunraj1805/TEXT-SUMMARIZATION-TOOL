# neuralstyle.py
import torch
from style_transfer_utils import load_image, run_style_transfer, im_convert
import torchvision.models as models
import matplotlib.pyplot as plt

# Paths to your images
content_path = "C:\\Users\\varun\\OneDrive\\Desktop\\neural_style_transfer\\god.jpg"
style_path = "C:\\Users\\varun\\OneDrive\\Desktop\\neural_style_transfer\\van_gogh.jpeg"

# Load images
content = load_image(content_path).to("cuda" if torch.cuda.is_available() else "cpu")
style = load_image(style_path, shape=content.shape[-2:])

# Load pre-trained VGG
vgg = models.vgg19(pretrained=True)

# Define layers for content and style
content_layers = ['21']  # relu4_2
style_layers = ['0', '5', '10', '19', '28']  # relu1_1, relu2_1, ...

# Run style transfer
output = run_style_transfer(vgg, content, style, content_layers, style_layers)

# Show output
plt.imshow(im_convert(output))
plt.axis('off')
plt.title("Stylized Image")
plt.show()
