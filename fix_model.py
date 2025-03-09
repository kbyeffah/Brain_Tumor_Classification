import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense

# Load the original model
old_model = load_model("2019-06-07_VGG_model.h5", compile=False)

# Create a new Sequential model
fixed_model = Sequential()

# Copy all layers except the last one
for layer in old_model.layers[:-1]:  
    fixed_model.add(layer)

# Freeze the pre-trained layers to retain learned features
for layer in fixed_model.layers:
    layer.trainable = False  

# Add a new output layer
fixed_model.add(Dense(4, activation="softmax"))

# Save the modified model
fixed_model.save("fixed_model.keras")

print("Model fixed and saved as 'fixed_model.keras'")
