import pandas as pd
import keras

# can test /255 or standardization
# or data augmentation or not
# i guess data augmentation is pretty low, so we can worry about it later

def separate_labels(data):
    data = pd.DataFrame(data)
    labels = data['label']
    data = data.drop(columns=['label'])
    return labels.to_numpy(), data.to_numpy()

def standardize(data):
    return data / 255.0

def train_test_val_split(data: pd.DataFrame, test_pct, val_pct):
    train, test = keras.utils.split_dataset(data.to_numpy(), right_size=test_pct)
    train, val = keras.utils.split_dataset(train, right_size=val_pct/(1-test_pct))
    return train.to_numpy(), val.to_numpy(), test.to_numpy()

def train_val_split(data, val_pct):
    train, val = keras.utils.split_dataset(data.to_numpy(), right_size=val_pct)
    return train, val

def preprocess_test_val(data, test_pct, val_pct):
    print("STANDARDIZE")
    data = standardize(data)
    print("TRAINTESTVALSPLIT")
    train, val, test = train_test_val_split(data, test_pct, val_pct)
    print("TRAIN SEP LABELS")
    train_y, train_x = separate_labels(train)
    print("VAL SEP LABELS")
    val_y, val_x = separate_labels(val)
    print("TEST SEP LABELS")
    test_y, test_x = separate_labels(test)
    return train_x, train_y, val_x, val_y, test_x, test_y

def preprocess_val(data, val_pct):
    data = standardize(data)
    train, val = train_val_split(data, val_pct)
    train_y, train_x = separate_labels(train)
    val_y, val_x = separate_labels(val)
    return train_x, train_y, val_x, val_y

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    trx, tr_y, valx, valy, tex, tey = preprocess_test_val(train, .2, .2)
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