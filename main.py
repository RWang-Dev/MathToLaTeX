import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
import random
import pickle
from functools import cmp_to_key
from tensorflow.keras.backend import clear_session

# Some symbols had to be filtered out since they were not well documented, conflicted with other symbols, or were beyond the scope of this project
# Symbols include: , | [ ] alpha ...

# A mapping between the default folder names and the corresponding latex syntax
# Not all of these were used, since the dataset was pruned
folder_to_latex = {
    '!': '!',
    '(': '\\left(',
    ')': '\\right)',
    '+': '+',
    ',': ',',
    '-': '-',
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '=': '=',
    'A': 'A',
    'alpha': '\\alpha',
    'ascii_124': '|',
    'b': 'b',
    'beta': '\\beta',
    'C': 'C',
    'cos': '\\cos',
    'd': 'd',
    'Delta': '\\Delta',
    'div': '\\div',
    'e': 'e',
    'exists': '\\exists',
    'f': 'f',
    'forall': '\\forall',
    'forward_slash': '/',
    'G': 'G',
    'gamma': '\\gamma',
    'geq': '\\geq',
    'gt': '>',
    'H': 'H',
    'i': 'i',
    'in': '\\in',
    'infty': '\\infty',
    'int': '\\int',
    'j': 'j',
    'k': 'k',
    'l': 'l',
    'lambda': '\\lambda',
    'ldots': '\\ldots',
    'leq': '\\leq',
    'lim': '\\lim',
    'log': '\\log',
    'lt': '<',
    'M': 'M',
    'mu': '\\mu',
    'N': 'N',
    'neq': '\\neq',
    'o': 'o',
    'p': 'p',
    'phi': '\\phi',
    'pi': '\\pi',
    'pm': '\\pm',
    'prime': "'",
    'q': 'q',
    'R': 'R',
    'rightarrow': '\\rightarrow',
    'S': 'S',
    'sigma': '\\sigma',
    'sin': '\\sin',
    'sqrt': '\\sqrt',
    'sum': '\\sum',
    'T': 'T',
    'tan': '\\tan',
    'theta': '\\theta',
    'times': '\\times',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'X': 'X',
    'y': 'y',
    'z': 'z',
    '[': '\\left[',
    ']': '\\right]',
    '{': '\\left\\{',
    '}': '\\right\\}',
}

# Loads and preprocesses images from the given dataset directory
# Returns the resized images, labels, and label_map
def load_data(data_dir, images_per_class, random_state=None):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    if random_state is not None:
        random.seed(random_state)

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            latex_label = folder_to_latex.get(folder_name, folder_name)
            label_map[current_label] = latex_label
            print(f"Processing folder: {folder_name} -> Label: {latex_label}")

            all_images = os.listdir(folder_path)
            random.shuffle(all_images)
            num_images_to_load = min(images_per_class, len(all_images))
            selected_images = all_images[:num_images_to_load]

            for img_name in selected_images:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (28, 28)) / 255.0
                images.append(img)
                labels.append(current_label)
            current_label += 1

    images = np.array(images).reshape(-1, 28, 28, 1)
    labels = np.array(labels)
    return images, labels, label_map

data_dir = 'data/extracted_images'
data_npz_path = 'processed_data.npz'
label_map_path = 'label_map.pkl'

if os.path.exists(data_npz_path) and os.path.exists(label_map_path):
    print("Loading processed data...")
    data = np.load(data_npz_path)
    images = data['images']
    labels = data['labels']
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
    print("Data loaded successfully.")
else:
    print("Processed data not found. Loading and processing images...")
    images, labels, label_map = load_data(
        data_dir,
        images_per_class=5000,
        random_state=42
    )
    np.savez(data_npz_path, images=images, labels=labels)
    with open(label_map_path, 'wb') as f:
        pickle.dump(label_map, f)
    print("Data processed and saved successfully.")

print("\nLabel Map:")
for key, value in label_map.items():
    print(f"{key}: {value}")

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42)

num_classes = len(label_map)
print(f"\nNumber of classes: {num_classes}")


model_keras_path = "character_recognizer.keras"

# We want to check if the model has been saved before training a new one
if os.path.exists(model_keras_path):
    print("Loading trained model...")
    model = models.load_model(model_keras_path)
    print("Model loaded successfully")
else:
    # We used a Convolutional Neural Network here which is good for character recognition
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),     # We experimented here with dropout layers which are supposed to help with overfitting, but more data helped as well
        layers.Dense(num_classes, activation='softmax')
    ])


    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    print("\nStarting model training...")
    history = model.fit(X_train, y_train, epochs=15, 
                       validation_data=(X_val, y_val))

    model.save(model_keras_path)
    print(f"\nModel saved as '{model_keras_path}'")
    clear_session()



# Function to segment the expression image into individual characters
# Basically for this function we want to identify the different characters in an equation and the sort them from left to right, alongside their bounding boxes
def segment_expression(image):
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    char_images = []

    for bbox in bounding_boxes:
        x, y, w, h = bbox
        char_img = image[y:y+h, x:x+w]
        char_img = resize_and_pad(char_img, size=28)
        char_img = char_img / 255.0
        char_images.append(char_img)
    return char_images

# After segmenting the images, it is helpful to resize the images to the size of the training images by padding the borders with white pixels
def resize_and_pad(image, size=28):
    h, w = image.shape
    scaling_factor = size / max(h, w)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    resized_image = cv2.resize(image, (new_w, new_h))
    pad_w = size - new_w
    pad_h = size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=255)  # Pad with white pixels
    return padded_image

# Same idea as the segmentation function above but it also draws bounding boxes to help with visualization
def segment_expression_with_visualization(image):
    
    imgGray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    imgCanny = cv2.Canny(imgGray, 50,180)
    kernel = np.ones((5,5), np.uint8)
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=2)

    contours, _= cv2.findContours(imgDilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)
        min_area = 20
        max_area = 10000
        is_possible_equals = aspect_ratio > 1.5 and w > 10
        
        if (min_area < area < max_area) or is_possible_equals:
            filtered_contours.append(contour)
    
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    char_images = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        aspect_ratio = w / float(h)
        color = (0, 255, 0)
        cv2.rectangle(imgGray, (x, y), (x + w, y + h), color, 2)
        
        char_img = image[y:y+h, x:x+w]
        char_img = resize_and_pad(char_img, size=28)
        char_img = char_img / 255.0
        char_images.append(char_img)

    cv2.imshow('Segmented Characters', imgGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return char_images


expression_image = cv2.imread('test_img8.jpg', cv2.IMREAD_GRAYSCALE)

if expression_image is None:
    print("Error: Expression image not found.")
else:
    char_images = segment_expression_with_visualization(expression_image)
    char_images_array = np.array(char_images).reshape(-1, 28, 28, 1)

    # Load the trained model
    predictions = model.predict(char_images_array)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_sequence = [label_map[label] for label in predicted_labels]

    def sequence_to_latex(sequence):

        latex_code = ''.join(sequence)
        return latex_code

    latex_output = sequence_to_latex(predicted_sequence)
    print("\nPredicted LaTeX Code:")
    print(latex_output)