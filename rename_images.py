import os

def rename_images_in_folder(folder_path, prefix):
    """
    Renames all images in a folder with a given prefix and sequential numbering.

    Args:
        folder_path (str): Path to the folder containing images.
        prefix (str): Prefix for the renamed images (e.g., 'ather' or 'other').
    """
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Filter to include only image files (jpg, png, etc.)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sort files to ensure consistent naming order
    image_files.sort()

    # Rename images
    for i, filename in enumerate(image_files, start=1):
        # Create new filename with sequential numbering
        new_name = f"{prefix}_{i:03d}.jpg"  # Example: ather_001.jpg
        
        # Get full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Paths to your folders
ather_folder = "dataset/ather_scooters"
other_folder = "dataset/motor_scooters"

# Rename images in each folder
rename_images_in_folder(ather_folder, "ather")
rename_images_in_folder(other_folder, "other")

print("Renaming completed!")
