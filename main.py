import data_loader
import model
import utils
import streamlit as st
from PIL import Image
import numpy as np

def predict_new_image(image, model_filename='model.keras'):
    try:
        b_model_loaded = model.load_saved_model(model_filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return "Error: Model not loaded", None

    image_to_predict = utils.preprocess_image(image)

    y_predicted_new = model.predict(b_model_loaded, image_to_predict)

    predicted_label = model.label_prediction(y_predicted_new[0])

    return predicted_label, y_predicted_new[0]

def run_streamlit_app():
    st.title("CIFAR-10 Image Classifier")



    st.header("2. Predict New Image")

    uploaded_file = st.file_uploader(
        "Upload a 32x32 image for prediction (e.g., JPEG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                predicted_label, raw_probs = predict_new_image(image, model_filename='model.keras')

                st.success(f"**Predicted Class:** {predicted_label}")
                st.subheader("Prediction Probabilities")
                class_names = [
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
                
                prob_data = {
                    "Class": class_names,
                    "Probability": [f"{p*100:.2f}%" for p in raw_probs]
                }
                st.dataframe(prob_data)
    else:
        st.info("Please upload an image to predict its class.")


@st.cache_data
def load_data_and_preprocess():
    X,y = data_loader.load_data()
    x_train, x_test, y_train, y_test = data_loader.split_data(X, y)
    x_train_norm, x_test_norm = data_loader.Normalize(x_train, x_test)
    return x_train, x_test, y_train, y_test, x_train_norm, x_test_norm

def main_train_and_evaluate():
    """Performs the full model training and evaluation pipeline."""
    x_train, x_test, y_train, y_test, x_train_norm, x_test_norm = load_data_and_preprocess()

    st.write(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    st.subheader("Model Summary")
    b_model = model.build_model()
    
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        b_model.summary()
    st.text(f.getvalue())

    st.subheader("Training Model")
    history = model.train_model(b_model, x_train_norm, y_train)

    st.subheader("Training History")
    st.pyplot(utils.plot_training(history))

    model.save_model(b_model)
    st.success("Model saved as 'model.keras'")

    st.subheader("Evaluation")
    test_loss, test_acc = b_model.evaluate(x_test_norm, y_test, verbose=0)
    st.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    y_predicted = model.predict(b_model, x_test_norm)

    st.subheader("Confusion Matrix")
    st.pyplot(utils.CM(y_test, y_predicted))

    st.subheader("Sample Misclassified Images")
    fig_mis = utils.mismisclassified_pre(x_test, y_test, y_predicted)
    st.pyplot(fig_mis)

    st.subheader("Sample Correctly Classified Images")
    fig_corr = utils.correct_pre(x_test, y_test, y_predicted)
    st.pyplot(fig_corr)


if __name__ == "__main__":
    run_streamlit_app()





