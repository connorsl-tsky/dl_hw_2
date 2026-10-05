import pandas as pd
import numpy as np
import sklearn.model_selection as sk

def separate_labels(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    data = pd.DataFrame(data)
    labels = data['label']
    data = data.drop(columns=['label'])
    return data.to_numpy(), labels.to_numpy() 

def standardize(data: pd.DataFrame) -> pd.DataFrame:
    return data / 255.0

def train_test_val_split(data: pd.DataFrame, test_pct: float, val_pct: float, shuffle: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = sk.train_test_split(data, test_size=test_pct, shuffle=shuffle)
    train, val = sk.train_test_split(train, test_size=val_pct/(1-test_pct), shuffle=shuffle)
    return train, val, test

def train_val_split(data: pd.DataFrame, val_pct: float, shuffle: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = sk.train_test_split(data, test_size=val_pct, shuffle=shuffle)
    return train, val

def reshape_for_conv(data: pd.DataFrame) -> pd.DataFrame:
    return np.reshape(data, (data.shape[0], 28, 28, 1))

def preprocess_test_val(data: pd.DataFrame, stdz: bool, test_pct: float, val_pct: float, shuffle: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, val, test = train_test_val_split(data=data, test_pct=test_pct, val_pct=val_pct, shuffle=shuffle)
    train_x, train_y = separate_labels(train)
    val_x, val_y = separate_labels(val)
    test_x, test_y = separate_labels(test)
    if stdz:
        train_x = standardize(train_x)
        val_x = standardize(val_x)
        test_x = standardize(test_x)
    return train_x, train_y, val_x, val_y, test_x, test_y

def preprocess_val(data: pd.DataFrame, stdz: bool, val_pct: float, shuffle: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, val = train_val_split(data=data, val_pct=val_pct, shuffle=shuffle)
    train_x, train_y = separate_labels(train)
    val_x, val_y = separate_labels(val)
    if stdz:
        train_x = standardize(train_x)
        val_x = standardize(val_x)
    return train_x, train_y, val_x, val_y

if __name__ == "__main__":
    data = pd.read_csv('train.csv')
    
    # dict = {'label': [11,12,13,14,15,16,17,18,19,110], "A": [1,2,3,4,5,6,7,8,9,10], "B": [1,2,3,4,5,6,7,8,9,10], "C": [1,2,3,4,5,6,7,8,9,10]}
    # data = pd.DataFrame(dict)



    trx, tr_y, valx, valy, tex, tey = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True)
    
    # trx, tr_y, valx, valy = preprocess_val(data, stdz=False, val_pct=.2, shuffle=True)
    
    
    
    print("\n\nTRAIN X")
    print(trx)
    print(trx.max())
    print(trx.shape)
    print("\n\nTRAIN Y")
    print(tr_y)
    print(tr_y.shape)
    print("\n\nVAL X")
    print(valx)
    print(valx.max())
    print(valx.shape)
    print("\n\nVAL Y")
    print(valy)
    print(valy.shape)
    print("\n\nTEST X")
    print(tex)
    print(tex.max())
    print(tex.shape)
    print("\n\nTEST Y")
    print(tey)
    print(tey.shape)