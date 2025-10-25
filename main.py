import data_loader
import model
import utils
import tkinter as tk
from tkinter import simpledialog, messagebox

def predict_new_image(image_path,model_filename='model.keras'):
    b_model_loaded = model.load_saved_model(model_filename)
    
    image_array = utils.preprocess_image(image_path)
    
    if image_array.shape[-1] == 1:
        image_to_predict = image_array.reshape(1, 32, 32, 3)
    else:
        image_to_predict = image_array
        
    print(f"Image array shape for prediction: {image_to_predict.shape}")

    y_predicted_new = model.predict(b_model_loaded, image_to_predict)

    predicted_label = model.label_prediction(y_predicted_new[0])

    return predicted_label, y_predicted_new[0]


def run_prediction_gui():

    root = tk.Tk()
    root.withdraw() 
    
    image_path = simpledialog.askstring(
        "Image Path Input", 
        "Please Enter the path of the image"
    )
    
    if image_path:
        result_label , raw_probs = predict_new_image(image_path, model_filename='model.keras')
        
        messagebox.showinfo(
                "Prediction Result", 
                f"Predicted Class: {result_label}"
            )
    else:
            messagebox.showerror("Error", result_label)

def main():
    df = data_loader.load_data()
    X = df.drop('label', axis=1).values.reshape(-1, 32, 32, 3)
    y = df['label'].values
    x_train, x_test, y_train, y_test = data_loader.split_data(X, y)
    print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    x_train_norm, x_test_norm = data_loader.Normalize(x_train, x_test)
    b_model = model.build_model()
    b_model.summary()
    history = model.train_model(b_model, x_train_norm, y_train)
    model.save_model(b_model)
    loaded_model_test = model.load_saved_model('model.keras')
    model.evaluate(b_model, x_test_norm, y_test)
    y_predicted = model.predict(b_model,x_test_norm)
    print("Plotting Confusion Matrix...")
    utils.CM(y_test, y_predicted)
    utils.mismisclassified_pre(x_test,y_test,y_predicted)
    utils.correct_pre(x_test,y_test,y_predicted)

if __name__ == "__main__":
    main()
    run_prediction_gui()