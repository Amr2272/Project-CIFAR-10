import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from PIL import Image

def plot_training(history):

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Loss')
    plt.legend()
    return plt.gcf()

def CM(y_test,y_predicted):
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    return plt.gcf()

def mismisclassified_pre(x_test,y_test,y_predicted):
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    misclassified_idx = np.where(y_predicted_labels != y_test)[0]

    print(f"Total Misclassified Images: {len(misclassified_idx)}")

    fig = plt.figure(figsize=(12, 10))
    for i, idx in enumerate(misclassified_idx[:15]):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[idx].astype('uint8')) 
        plt.title(f"True: {y_test[idx]}, Pred: {y_predicted_labels[idx]}", fontsize=10)
        plt.axis('off')
    plt.suptitle("Sample Misclassified Images", fontsize=16, fontweight="bold")
    return fig

def correct_pre(x_test,y_test,y_predicted):
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    correct_idx = np.where(y_predicted_labels == y_test)[0]
    print(f"Total Correctly Classified Images: {len(correct_idx)}")

    fig = plt.figure(figsize=(12, 10))
    for i, idx in enumerate(correct_idx[:15]):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[idx].astype('uint8')) 
        plt.title(f"True: {y_test[idx]}, Pred: {y_predicted_labels[idx]}", fontsize=10, color='green')
        plt.axis('off')
    plt.suptitle("Sample Correctly Classified Images", fontsize=16, fontweight="bold")
    return fig

def preprocess_image(image):

    image = image.resize((32, 32)) 
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
        
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
