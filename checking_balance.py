import os

# Define paths to the dataset folders
ather_folder = 'dataset/ather_scooters'
other_folder = 'dataset/motors_scooters'

# Count the number of images in each folder
ather_images = [f for f in os.listdir(ather_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
other_images = [f for f in os.listdir(other_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Print counts
print(f"Ather Scooter Images: {len(ather_images)}")
print(f"Other Scooter Images: {len(other_images)}")

# Check for imbalance
if len(other_images) != 0:
    imbalance_ratio = len(ather_images) / len(other_images)
    print(f"Imbalance Ratio (Ather:Other): {imbalance_ratio:.2f}")
else:
    print("No images found in 'other scooters' folder!")
