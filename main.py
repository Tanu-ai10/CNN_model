from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np


# Load the model from the saved file
model = load_model(r"C:\Users\lenovo\Desktop\coding\AI_Art_Detector\best_model.keras")

print("Model loaded successfully.")

# Define labels (must match your training labels)
label=['AI','Human'] # Replace with actual labels

# Image path for testing
image_path = r"C:\Users\lenovo\Desktop\coding\AI_Art_Detector\images\test\AI_LD_baroque\1-1900923-343813.jpg"
print("The original image is of Human")
# Preprocess the image
img = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size=(32, 32))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

# Predict and display results
pred = model.predict(img_array)
pred_label = label[np.argmax(pred)]
confidence = np.max(pred)  # Get the highest confidence score

print("Model prediction is:", pred_label)
print(f"Model prediction is: {pred_label} with confidence: {confidence:.2f}")
