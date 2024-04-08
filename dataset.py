data_directory = '/notebooks/data_dir/FER-2013'

import os
import shutil
import random

# Define paths
train_directory = os.path.join(data_directory, "train")
test_directory = os.path.join(data_directory, "test")
val_directory = os.path.join(data_directory, "val")

# Create validation directory if it doesn't exist
if not os.path.exists(val_directory):
    os.makedirs(val_directory)

# List the subdirectories (classes) in the train directory
classes = os.listdir(train_directory)

# Create subdirectories in the validation directory
for class_name in classes:
    class_val_directory = os.path.join(val_directory, class_name)
    if not os.path.exists(class_val_directory):
        os.makedirs(class_val_directory)

# Move images to the validation directory
for class_name in classes:
    class_train_directory = os.path.join(train_directory, class_name)
    class_val_directory = os.path.join(val_directory, class_name)
    
    # List files in the class directory
    files = os.listdir(class_train_directory)
    
    # Define the ratio for splitting (7:1)
    val_ratio = 0.125  # 1/8 of the data for validation
    
    # Calculate the number of images for validation
    num_val_images = int(val_ratio * len(files))
    
    # Randomly select images for validation
    val_files = random.sample(files, num_val_images)
    
    # Move validation set images to the validation directory
    for image_filename in val_files:
        image_path = os.path.join(class_train_directory, image_filename)
        shutil.move(image_path, os.path.join(class_val_directory, image_filename))

