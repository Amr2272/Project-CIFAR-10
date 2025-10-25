import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk

def plot_training(history):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.show()

def CM(y_test,y_predicted):
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')    
    plt.ylabel('Truth')
    plt.show()

def mismisclassified_pre(x_test,y_test,y_predicted):
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    misclassified_idx = np.where(y_predicted_labels != y_test)[0]

    print(f"Total Misclassified Images: {len(misclassified_idx)}")

    plt.figure(figsize=(12, 10))
    for i, idx in enumerate(misclassified_idx[:15]):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[idx].reshape(32,32,3))
        plt.title(f"True: {y_test[idx]}, Pred: {y_predicted_labels[idx]}", fontsize=10)
        plt.axis('off')
    plt.suptitle("Sample Misclassified Images", fontsize=16, fontweight="bold")
    plt.show()

def correct_pre(x_test,y_test,y_predicted):
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    correct_idx = np.where(y_predicted_labels == y_test)[0]
    print(f"Total Correctly Classified Images: {len(correct_idx)}")

    plt.figure(figsize=(12, 10))
    for i, idx in enumerate(correct_idx[:15]):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[idx])
        plt.title(f"True: {y_test[idx]}, Pred: {y_predicted_labels[idx]}", fontsize=10, color='green')
        plt.axis('off')
    plt.suptitle("Sample Correctly Classified Images", fontsize=16, fontweight="bold")
    plt.show()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32)) 
    img_array = np.array(img)
    img_array = img_array / 255
    img_array = img_array.reshape(1, 32, 32, 3)  
    return img_array