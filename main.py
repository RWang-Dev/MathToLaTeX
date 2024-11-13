import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
import random
import pickle

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

def load_data(data_dir, images_per_class=500, random_state=None):
    """
    Loads and preprocesses images from the dataset directory.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        images (np.ndarray): Array of image data.
        labels (np.ndarray): Array of labels corresponding to images.
        label_map (dict): Mapping from label indices to LaTeX commands.
    """
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
        images_per_class=500,
        random_state=42
    )
    np.savez(data_npz_path, images=images, labels=labels)
    with open(label_map_path, 'wb') as f:
        pickle.dump(label_map, f)
    print("Data processed and saved successfully.")

print("\nLabel Map:")
for key, value in label_map.items():
    print(f"{key}: {value}")
