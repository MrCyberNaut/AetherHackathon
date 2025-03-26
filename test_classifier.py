import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('ather_classifier_model.h5')

def predict_image(img_path):
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict using the model
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        print(f"{img_path}: Ather Scooter detected!")
        return "Ather Scooter"
    else:
        print(f"{img_path}: Not an Ather Scooter.")
        return "Other Scooter"

# Test on an example image (replace with your test image path)
test_image_path = "dataset/test_data/test01.jpeg"
result = predict_image(test_image_path)
print("Result:", result)

