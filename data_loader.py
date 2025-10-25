import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).flatten()
    
    return X,y

def Labels(y):
    print("Number of Lables : ",len(np.unique(y)))

def split_data(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  x_train, x_test, y_train, y_test

def Normalize(x_train, x_test):
    mx=x_train.max()
    x_train=x_train/mx
    x_test=x_test/mx
    return x_train,x_test





