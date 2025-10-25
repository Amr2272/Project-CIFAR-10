from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

def build_model():
    model = models.Sequential([

        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=70):
    history=model.fit(x_train, y_train, epochs=epochs)
    return history

def evaluate(model,x_test,y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\n Test accuracy: {test_acc:.4f}", "Test loss : ",test_loss)

def predict(model,x_test):
    y_predicted = model.predict(x_test)
    return y_predicted

def label_prediction(prediction_array):
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]
    predicted_class = prediction_array.argmax()
    return class_names[predicted_class]

def save_model(model, filename='model.keras'):
    model.save(filename)

def load_saved_model(filename='model.keras'):
    loaded_model = load_model(filename)
    return loaded_model


