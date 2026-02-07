# CIFAR-10 Image Classifier (Streamlit)

A Streamlit app for classifying CIFAR-10 images with a small CNN built in TensorFlow/Keras. It loads a saved model (`model.keras`) and predicts the class of an uploaded 32x32 image.

## Features
- Upload an image and get predicted class + probability table.
- CNN model definition and training utilities.
- Visualization helpers for training curves, confusion matrix, and sample predictions.

## Project Structure
- `main.py` - Streamlit app and prediction flow.
- `model.py` - CNN architecture, training, save/load helpers.
- `data_loader.py` - CIFAR-10 loading, splitting, normalization.
- `utils.py` - Plotting and image preprocessing.
- `model.keras` - Saved model weights.

## Setup
1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the App

```bash
streamlit run main.py
```

Open the URL shown in the terminal, upload a 32x32 image, and click **Predict**.

## Classes
The model predicts one of these CIFAR-10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Optional: Train and Evaluate
The training pipeline is implemented in `main_train_and_evaluate()` inside `main.py`. If you want to train and evaluate in Streamlit, call that function instead of `run_streamlit_app()` under `if __name__ == "__main__":`.

## Notes
- CIFAR-10 data is downloaded automatically by TensorFlow if not present.
- Uploaded images are resized to 32x32 and normalized to [0, 1].
