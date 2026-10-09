"""
https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

yeah just follow this guide. it's a special kind of CNN
"""

import keras
import pandas as pd
from Preprocessing import preprocess_test_val, reshape_for_conv
import numpy as np
import tensorflow as tf
from test import export_stats
import copy

class VGG:
    def __init__(self, optimizer: keras.Optimizer, loss: keras.Loss, metrics: list[keras.Metric], batch: int, epochs: int, early_stop: bool, patience: int | None, lrs: bool, scheduler, regularizer: keras.Regularizer):
        # member vars
        self.batch = batch
        self.epochs = epochs
        self.regularizer = regularizer

        # callbacks
        self.callbacks = []
        if early_stop and patience:
            early_stop_callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose = 1
            )
            self.callbacks.append(early_stop_callback)
        else:
            print("VGG: NO EARLY STOP OR PATIENCE")
        if lrs and scheduler:
            lrs_callback = keras.callbacks.LearningRateScheduler(
                schedule = scheduler,
                verbose=1
            )
            self.callbacks.append(lrs_callback)
        else:
            print("VGG: NO LEARNING RATE SCHEDULER OR SCHEDULER")

        # build model
        self.model = keras.models.Sequential()
        self.init_layers()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return

    def init_layers(self):
        """
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        """
        # block 1
        self.model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

        # block 2
        self.model.add(keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

        # block 3
        self.model.add(keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

        # block 4
        self.model.add(keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


        # block 5
        self.model.add(keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same'))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))


        # FC
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=4096,activation='relu', kernel_regularizer=self.regularizer))
        self.model.add(keras.layers.Dense(units=4096,activation='relu', kernel_regularizer=self.regularizer))
        self.model.add(keras.layers.Dense(units=1000,activation='softmax',))
        
        self.model.add(keras.layers.Dense(units=10)) # 10 for the digits
        return
    
    def train(self, train_x: np.ndarray, train_y, val_x: np.ndarray, val_y):
        self.loss = self.model.fit(
            train_x, 
            train_y, 
            batch_size=self.batch, 
            epochs=self.epochs, 
            validation_data=(val_x, val_y),
            verbose=2,
            callbacks=self.callbacks
        )
        return self.loss

    def evaluate(self, test_x: np.ndarray, test_y, callbacks: bool=False):
        callback = []
        if callbacks:
            callback = self.callbacks
        return self.model.evaluate(test_x, test_y, callbacks=callback)
        
    


def build_model(params: dict, i: int):
    """
    params = {
        "learning_rate": 0.001,
        "optimizer": {
            "name": "adam",
            "optimizer": keras.optimizers.Adam(
                learning_rate = 0.001,
            ),
        },
        "loss": {
            "name": "cce",
            "loss": keras.losses.CategoricalCrossentropy(),
        },
        "scheduler": {
            "name": "no change",
            "scheduler": lambda epoch, loss : loss,
        },
        "regularizer": {
            "name": "l2",
            "regularizer": keras.regularizers.l2()
        },
        "batch_size": 1024,
        "epochs": 400,
        "patience": 15,
    }
    """

    optimizer = assign_params_nested(params, "optimizer", i)
    loss = assign_params_nested(params, "loss", i)
    batch = assign_param(params, "batch_size", i)
    epochs = assign_param(params, "epochs", i)
    patience = assign_param(params, "patience", i)
    scheduler = assign_params_nested(params, "scheduler", i)
    regularizer = assign_params_nested(params, "regularizer", i)

    vgg = VGG(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.AUC(),
            keras.metrics.F1Score(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            # keras.metrics.IoU(),
        ],
        batch=batch,
        epochs=epochs,
        early_stop=True,
        patience=patience,
        lrs=True,
        scheduler=scheduler,
        regularizer=regularizer
    )

    return vgg


def assign_param(params: dict, key: str, i: int):
    """ 
    see build_model
    """
    if isinstance(params[key], list):
        return params[key][i]
    else:
        return params[key]

def assign_params_nested(params: dict, key: str, i: int):
    """
    see build_model
    """
    if isinstance(params[key], list):
        return params[key][i][key]
    else:
        return params[key][key]

def input_params_to_output_params(params: dict, i: int):
    out = copy.deepcopy(params)
    for k, v in out.items():
        if isinstance(out[k], list):
            out[k] = out[k][i]
        if isinstance(out[k], dict):
            out[k] = out[k]["name"]
    return out

def test_param():
    data = pd.read_csv("train.csv")
    random=1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    train_x = reshape_for_conv(train_x)
    val_x = reshape_for_conv(val_x)
    test_x = reshape_for_conv(test_x)

    train_x = tf.image.resize(train_x, (32, 32)).numpy()
    val_x   = tf.image.resize(val_x,   (32, 32)).numpy()
    test_x  = tf.image.resize(test_x,  (32, 32)).numpy()
    

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y
    }

    params = {
        "learning_rate": 0.001,
        "optimizer": {
            "name": "adam",
            "optimizer": keras.optimizers.Adam(
                learning_rate = 0.001,
            ),
        },
        "loss": {
            "name": "cce",
            "loss": keras.losses.CategoricalCrossentropy(),
        },
        "scheduler": {
            "name": "no change",
            "scheduler": lambda epoch, loss : loss,
        },
        "regularizer": {
            "name": "l2",
            "regularizer": keras.regularizers.l2()
        },
        "batch_size": 1024,
        "epochs": 400,
        "patience": 15,
    }

    
    test_key = "filters"

    for i in range(len(params[test_key])):
        cnn = build_model(params)
        output_params = input_params_to_output_params(params)
        export_stats(model=cnn, name=f"CNN_FILTER_{params[test_key][i]}", params=output_params, num_runs=5, data=data, random=random, output_precision=5)
    return


if __name__ == "__main__":
    print("hello world")