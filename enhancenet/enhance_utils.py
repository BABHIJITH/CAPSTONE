import torch
from enhancenet.model import EnhanceNet
from torchvision import transforms
from PIL import Image
import os

def enhance_signature(image_path):
    weights_path = 'models/enhancenet_weights.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{weights_path}' not found.")

    model = EnhanceNet()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        enhanced_image = model(image)

    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image = (enhanced_image * 255).astype('uint8')
    return Image.fromarray(enhanced_image)
