import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models

# Set the path to the CASIA Gait Dataset A
dataset_path = "C:\\Users\\agraw\\OneDrive\\Documents\\GaitDatasetA-silh"

# Helper function to extract label from folder name
def get_label(folder_name):
    label = 1  # Assign a constant label value of 1 for all samples (representing humans)
    return label


# Preprocess dataset and split into training and testing sets
def preprocess_dataset():
    images = []
    labels = []

    for subject_folder in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject_folder)
        
        if not os.path.isdir(subject_path):
            continue

        for sequence_folder in os.listdir(subject_path):
            sequence_path = os.path.join(subject_path, sequence_folder)
            if not os.path.isdir(sequence_path):
                continue

            for image_file in os.listdir(sequence_path):
                if image_file.endswith(".png"): 
                    image_path = os.path.join(sequence_path, image_file)
                    
                    # Extract the label from the subject folder name
                    label = get_label(sequence_folder)
                    
                    # Load and preprocess image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (64, 128))  # Adjust size as needed

                    images.append(image)
                    labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Define the architecture of the model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Preprocess the dataset
X_train, X_test, y_train, y_test = preprocess_dataset()

# Reshape the data for the model
X_train = X_train.reshape(X_train.shape[0], 128, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 64, 1)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create the model
model = create_model()

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("gait_detection_model.h5")

# Save the X_test and y_test data
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
