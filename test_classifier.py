import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

# Load the trained model
model = load_model('ather_classifier_model.h5')

# Paths to test dataset and individual test image
test_dataset_path = 'dataset/test_data'  # Update this to your test dataset folder path
test_image_path = 'dataset/test_data/test01.jpg'  # Update this to your test image path

# Evaluate model on test dataset
def evaluate_model_on_test_data(test_dataset_path):
    # Prepare test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False  # Ensure consistent order for evaluation
    )

    # Get true labels and predictions
    y_true = test_generator.classes
    y_pred = (model.predict(test_generator) > 0.5).astype("int32")

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Predict single image classification result
def predict_single_image(img_path):
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Error: Unable to read image at {img_path}. Check file path or integrity.")
    
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Predict using the model
    prediction = model.predict(img_array)[0][0]
    
    label = "Ather Scooter" if prediction > 0.5 else "Other Scooter"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\nPrediction for {img_path}:")
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2f}")

# Evaluate model on test dataset
evaluate_model_on_test_data(test_dataset_path)

# Predict single image classification result
predict_single_image(test_image_path)
