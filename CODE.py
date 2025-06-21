import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import torch.nn.functional as F

# Create output directories
os.makedirs('masks', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Load pretrained DINO model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_path, n_clusters=5, pca_components=3):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Extract patch embeddings
    with torch.no_grad():
        features = model.get_intermediate_layers(image_tensor, n=1)[0]
    
    # Process features
    features = features[0, 1:, :].cpu().numpy()
    h = w = int(np.sqrt(features.shape[0]))
    
    # Apply PCA
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(features)
    reduced_features = reduced_features.reshape(h, w, -1)
    
    # K-means clustering with explicit n_init
    flattened_features = reduced_features.reshape(-1, pca_components)
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,  
        random_state=0
    ).fit(flattened_features)
    labels = kmeans.labels_.reshape(h, w)
    
    # Create mask
    labels_tensor = torch.tensor(labels).unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(labels_tensor, size=(224, 224), mode='nearest').squeeze().numpy()
    
    return image, mask

def save_results(original, mask, base_name):
    plt.imsave(f'masks/{base_name}_mask.png', mask, cmap='nipy_spectral')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(mask, cmap='nipy_spectral')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    plt.savefig(f'visualizations/{base_name}_comparison.png', bbox_inches='tight')
    plt.close()

# Process images in current directory
image_dir = os.getcwd()
for img_file in os.listdir(image_dir):
    if img_file.endswith('.png'):
        img_path = os.path.join(image_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        try:
            image, mask = process_image(img_path)
            save_results(image.resize((224, 224)), mask, base_name)
            print(f"Processed: {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")