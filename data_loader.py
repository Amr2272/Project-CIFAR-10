import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

def load_data():
    df=pd.read_csv(r'C:\Users\Amr\Desktop\Project  CIFAR-10\Dataset\CIFAR-10.csv')

    X = df.drop('label', axis=1).values
    y = df['label'].values
    X = X.reshape(-1, 32, 32,3)
    return df

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
