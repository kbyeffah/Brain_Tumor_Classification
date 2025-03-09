import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define image size & batch size
IMG_SIZE = (224, 224)  # Resize to 224x224
BATCH_SIZE = 32

# Define paths to the dataset
train_dir = "sample_data/train"
test_dir = "sample_data/test"

# Apply Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values (0-1)
    rotation_range=20,      # Rotate images randomly
    width_shift_range=0.2,  # Shift width
    height_shift_range=0.2, # Shift height
    shear_range=0.2,        # Shear transformation
    zoom_range=0.2,         # Zoom-in/out
    horizontal_flip=True,   # Flip images horizontally
    validation_split=0.2    # 20% for validation
)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Use 'binary' for two classes
    subset='training'  # Training set
)

# Load Validation Data
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Validation set
)

# Load Test Data (without augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Data preprocessing complete!")
