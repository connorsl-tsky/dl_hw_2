import keras
import pandas as pd
from Preprocessing import preprocess_test_val, reshape_for_conv
import numpy as np
from test import export_stats
import copy
import tensorflow as tf

class CNN:
    def __init__(self, filters: int, kernel_size: tuple[int], strides: tuple[int], padding: str, activation: str, pool_size: tuple[int], dense_units: int, optimizer: keras.Optimizer, loss: keras.Loss, metrics: list[keras.Metric], batch: int, epochs: int, early_stop: bool, patience: int | None, regularizer: keras.Regularizer, lrs: bool, scheduler):
        # member vars
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding # either 'valid' (no padding) or 'same'
        self.pool_size = pool_size
        self.dense_units = dense_units # num of nodes for the non-output dense layer
        self.activation = activation
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
            print("CNN: NO EARLY STOP OR PATIENCE")
        if lrs and scheduler:
            lrs_callback = keras.callbacks.LearningRateScheduler(
                schedule = scheduler,
                verbose=1
            )
            self.callbacks.append(lrs_callback)
        else:
            print("CNN: NO LEARNING RATE SCHEDULER OR SCHEDULER")

        # build model
        self.model = keras.models.Sequential()
        self.init_layers()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return

    def init_layers(self):
        self.model.add(
            keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                activation=self.activation
            )
        )
        self.model.add(
            keras.layers.MaxPool2D(
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding
            )
        )
        self.model.add(
            keras.layers.Flatten()
        )
        self.model.add(
            keras.layers.Dense(
                units=self.dense_units,
                activation=self.activation,
                kernel_regularizer=self.regularizer
            )
        )
        self.model.add(keras.layers.Dense(units=10, activation="softmax")) # 10 for the digits
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
        
    


def test_filters():
    filters_list = [4, 8, 16, 32, 64, 128]

    data = pd.read_csv("train.csv")
    random=1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    train_x = reshape_for_conv(train_x)
    val_x = reshape_for_conv(val_x)
    test_x = reshape_for_conv(test_x)

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y
    }

    for i in range(len(filters_list)):
        learning_rate = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        adam = {
            "name": "adam",
            "opt": keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2
            )
        }

        cce = {
            "name": "cce",
            "loss": keras.losses.CategoricalCrossentropy()
        }

        def scheduler(epoch, loss): 
            return loss
        
        sched = {
            "name": "direct loss",
            "scheduler": scheduler
        }

        l2_val = 0.01
        l2 = {
            "name": "l1",
            "val": l2_val,
            "reg": keras.regularizers.l2(l2_val)
        }

        filters=filters_list[i]
        kernel_size=(3,3)
        strides=(1,1)
        padding="same"
        pool_size=(3,3)
        dense_units=128
        batch=1024
        epochs=400
        patience=15
        cnn = CNN(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation="relu",
            pool_size=pool_size,
            dense_units=dense_units,
            optimizer=adam["opt"],
            loss=cce["loss"],
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
            scheduler=sched["scheduler"],
            regularizer=l2["reg"]
        )

        params = {
            "learning_rate": learning_rate,
            "optimizer": adam["name"],
            "loss func": cce["name"],
            "scheduler": sched["name"],
            "regularizer": l2["name"],
            "filters": filters_list[i],
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": padding,
            "pool_size": pool_size,
            "dense units": dense_units,
            "batch_size": batch,
            "epochs": epochs,
            "patience": patience,
        }

        export_stats(model=cnn, name=f"CNN_FILTER_{filters_list[i]}", params=params, num_runs=5, data=data, random=random)
    return

def build_model(params: dict, i: int):
    """
    params = {
        "learning_rate": 0.001,
        "optimizer": {
            "name": adam,
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
            "scheduler": scheduler,
        },
        "regularizer": {
            "name": l2,
            "regularizer": keras.regularizers.l2()
        },
        "filters": [4,8,16,32,64],
        "kernel_size": (3,3),
        "strides": (1,1),
        "padding": "same",
        "pool_size": (3,3),
        "dense_units": 128,
        "batch_size": 1024,
        "epochs": 400,
        "patience": 15,
    }
    """

    filters = assign_param(params, "filters", i)
    kernel_size = assign_param(params, "kernel_size", i)
    strides = assign_param(params, "strides", i)
    padding = assign_param(params, "padding", i)
    pool_size = assign_param(params, "pool_size", i)
    dense_units = assign_param(params, "dense_units", i)
    optimizer = assign_params_nested(params, "optimizer", i)
    loss = assign_params_nested(params, "loss", i)
    batch = assign_param(params, "batch_size", i)
    epochs = assign_param(params, "epochs", i)
    patience = assign_param(params, "patience", i)
    scheduler = assign_params_nested(params, "scheduler", i)
    regularizer = assign_params_nested(params, "regularizer", i)

    cnn = CNN(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation="relu",
        pool_size=pool_size,
        dense_units=dense_units,
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

    return cnn


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
    print(params)
    print(key)
    print(i)
    if isinstance(params[key], list):
        return copy.deepcopy(params[key][i][key])
    else:
        return copy.deepcopy(params[key][key])

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
        "filters": 16,
        "kernel_size": [(4,4), (4,5), (5,4), (5,5), (6,6), (7,7)],
        "strides": (1,1),
        "padding": "same",
        "pool_size": (3,3),
        "dense_units": 128,
        "batch_size": 1024,
        "epochs": 400,
        "patience": 15,
    }

    test_key = "kernel_size"

    for i in range(len(params[test_key])):
        cnn = build_model(params, i)
        output_params = input_params_to_output_params(params, i)
        export_stats(model=cnn, name=f"CNN_{test_key}_{params[test_key][i]}", params=output_params, num_runs=5, data=data, random=random, output_precision=5)
    return

def clear_optimizer(optimizer: keras.Optimizer):
    for var in optimizer.variables:
        var.assign(tf.zeros_like(var))
    return optimizer


"""
testing list
 - filters
 16 is best
 - kernel_size
 - stride
 - padding
 - pool size
 - dense units
 everything else should be pretty good
"""

if __name__ == "__main__":
    test_param()