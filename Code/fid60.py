import os
import csv
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define the function to calculate the FID score
def calculate_fid_score(img_path1, img_path2, model):
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the images
    img1 = transform(Image.open(img_path1).convert('RGB')).unsqueeze(0)
    img2 = transform(Image.open(img_path2).convert('RGB')).unsqueeze(0)

    # Calculate the activations for both images
    with torch.no_grad():
        activations1 = model(img1)[0].squeeze()
        activations2 = model(img2)[0].squeeze()

    # Calculate the means and variances for the two sets of activations
    mu1 = torch.mean(activations1, dim=0)
    mu2 = torch.mean(activations2, dim=0)
    sigma1 = torch.var(activations1, dim=0, unbiased=False)
    sigma2 = torch.var(activations2, dim=0, unbiased=False)

    # Calculate the FID score using the means and variances
    fid = torch.norm(mu1 - mu2, 2) + torch.norm(sigma1 - sigma2, 2)

    return fid.item()

# Load the Inception model with the recommended 'weights' parameter
inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).eval()

# Path to the main directory and subdirectory
main_dir = './'
sub_dir = './aigraph/'

# List to store the FID scores
fid_scores = []

# Filename for the reference image
reference_image_path = os.path.join(main_dir, '0.png')

# Get the list of image files in the subdirectory
image_files = [f for f in os.listdir(sub_dir) if f.endswith('.png')]

# Calculate the FID score for each image in the subdirectory
for image_file in image_files:
    img_path2 = os.path.join(sub_dir, image_file)
    fid_score = calculate_fid_score(reference_image_path, img_path2, inception_model)
    fid_scores.append(fid_score)

# Save the FID scores to a CSV file
csv_file_path = 'fid_scores.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Filename', 'FID Score'])
    for image_file, fid_score in zip(image_files, fid_scores):
        writer.writerow([image_file, fid_score])


