import torch
from enhancenet.model import EnhanceNet
from torchvision import transforms
from PIL import Image

def enhance_signature(image_path):
    # Load pre-trained model
    model = EnhanceNet()
    model.load_state_dict(torch.load('models/enhancenet_weights.pth'))
    model.eval()

    # Preprocess input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Enhance the image
    with torch.no_grad():
        enhanced_image = model(image)
    
    # Convert enhanced image back to PIL format
    enhanced_image = enhanced_image.squeeze(0).permute(1, 2, 0).numpy()
    enhanced_image = (enhanced_image * 255).astype('uint8')  # Convert to 0-255 range
    return Image.fromarray(enhanced_image)
