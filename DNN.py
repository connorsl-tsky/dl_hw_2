
import keras
import pandas as pd
from Preprocessing import preprocess_test_val, preprocess_val, standardize
import numpy as np
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from test import export_stats
import math
import copy

class DNN:
    def __init__(self, layers: list[int], activation: str, optimizer: keras.Optimizer, loss: keras.Loss, metrics: list[keras.Metric], batch: int, epochs: int, early_stop: bool, patience: int | None, regularizer: keras.Regularizer, lrs: bool, scheduler):
        # member vars
        self.layers = layers
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
            print("DNN: NO EARLY STOP OR PATIENCE")
        if lrs and scheduler:
            lrs_callback = keras.callbacks.LearningRateScheduler(
                schedule = scheduler,
                verbose=1
            )
            self.callbacks.append(lrs_callback)
        else:
            print("DNN: NO LEARNING RATE SCHEDULER OR SCHEDULER")

        # build model
        self.model = keras.models.Sequential()
        self.init_layers()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return

    def init_layers(self):
        for layer in self.layers:
            self.model.add(
                keras.layers.Dense(
                    units=layer, 
                    activation=self.activation, 
                    kernel_regularizer=self.regularizer
                )
            )
        self.model.add(keras.layers.Dense(units=10, activation="softmax")) # 10 for the digits
        return
    
    def train(self, train_x, train_y, val_x, val_y):
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

    def evaluate(self, test_x, test_y, callbacks: bool=False):
        callback = []
        if callbacks:
            callback = self.callbacks
        return self.model.evaluate(test_x, test_y, callbacks=callback)
        
    
def test_learning_rate():
    # learning_rate_list = [0.9, 0.1, 0.01, 0.001, 0.0001, 0.00001] # error on 0.001
    # learning_rate_list = [0.99, 0.999, 0.9999]
    # learning_rate_list = [0.01, 0.001, 0.0001]
    learning_rate_list = [0.001]

    data = pd.read_csv("train.csv")
    random_state = 1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random_state)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        beta_1 = 0.9
        beta_2 = 0.999
        adam = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
    
        cce = keras.losses.CategoricalCrossentropy()
    
        def scheduler(epoch, loss): 
            return loss
    
        l2_val = 0.01
        l2 = keras.regularizers.l2(l2_val)
    
        layers=[64]
        batch=1024
        epochs=200
        patience=15
        dnn = DNN(
            layers=layers,
            activation="relu",
            optimizer=adam,
            loss=cce,
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.F1Score(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],
            batch=batch,
            epochs=epochs,
            early_stop=True,
            patience=patience,
            lrs=True,
            scheduler=scheduler,
            regularizer=l2
        )
        params = {
            "optimizer": "adam",
            "learning rate": learning_rate,
            "beta_1": beta_1,
            "beta 2": beta_2,
            "loss": "categorical cross entropy",
            "lrs scheduler": "none",
            "regularizer": "l2",
            "l2": l2_val,
            "layers": layers,
            "batch": batch,
            "epochs": epochs,
            "patience": patience
        }

        data = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y
        }

        export_stats(model=dnn, name=f"DNN_LEARNING_RATE_2_{learning_rate_list[i]}", params=params, num_runs=5, data=data, random=random_state)
    return 

def test_layers():
    # layer_list = [[256],[512],[1024],[2048],[4096]]
    # layer_list = [[512, 256, 128],[1024, 512, 256],[2048, 1024, 512],[4096, 2048, 1024],[4096, 4096, 1024]]
    # layer_list=[[128],[64],[32],[16],[8],[4]]
    # layer_list=[[4,4],[8,8],[16,16],[32,32],[64,64],[128,128],[256,256],[512,512],[1024,1024],[2048,2048],[4096,4096]]
    layer_list=[[16,8],[32,16],[64,32],[128,64],[256,128],[512,256]]

    data = pd.read_csv("train.csv")
    random_state = 1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random_state)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    for i in range(len(layer_list)):
        # learning_rate = 0.9999
        learning_rate = 0.01
        beta_1 = 0.9
        beta_2 = 0.999
        adam = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
    
        cce = keras.losses.CategoricalCrossentropy()
    
        def scheduler(epoch, loss): 
            return loss
    
        l1_val = 0.01
        l1 = keras.regularizers.l1(l1_val)
    
        layers=layer_list[i]
        batch=1024
        epochs=100
        patience=15
        activation="relu"
        dnn = DNN(
            layers=layers,
            activation=activation,
            optimizer=adam,
            loss=cce,
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.F1Score(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],
            batch=batch,
            epochs=epochs,
            early_stop=True,
            patience=patience,
            lrs=True,
            scheduler=scheduler,
            regularizer=l1
        )
        params = {
            "optimizer": "adam",
            "learning rate": learning_rate,
            "beta_1": beta_1,
            "beta 2": beta_2,
            "loss": "categorical cross entropy",
            "lrs scheduler": "none",
            "regularizer": "l1",
            "l1": l1_val,
            "layers": layers,
            "batch": batch,
            "epochs": epochs,
            "patience": patience,
            "activation": activation
        }

        data = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y
        }

        export_stats(model=dnn, name=f"DNN_LAYER_{layer_list[i]}", params=params, num_runs=5, data=data, random=random_state)
    return 

def test_batches():
    batch_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    data = pd.read_csv("train.csv")
    random_state = 1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random_state)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    for i in range(len(batch_list)):
        # learning_rate = 0.9999
        learning_rate = 0.01
        beta_1 = 0.9
        beta_2 = 0.999
        adam = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
    
        cce = keras.losses.CategoricalCrossentropy()
    
        def scheduler(epoch, loss): 
            return loss
    
        l1_val = 0.01
        l1 = keras.regularizers.l1(l1_val)
    
        layers=[64]
        batch=batch_list[i]
        epochs=100
        patience=15
        activation="relu"
        dnn = DNN(
            layers=layers,
            activation=activation,
            optimizer=adam,
            loss=cce,
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.F1Score(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],
            batch=batch,
            epochs=epochs,
            early_stop=True,
            patience=patience,
            lrs=True,
            scheduler=scheduler,
            regularizer=l1
        )
        params = {
            "optimizer": "adam",
            "learning rate": learning_rate,
            "beta_1": beta_1,
            "beta 2": beta_2,
            "loss": "categorical cross entropy",
            "lrs scheduler": "none",
            "regularizer": "l1",
            "l1": l1_val,
            "layers": layers,
            "batch": batch,
            "epochs": epochs,
            "patience": patience,
            "activation": activation
        }

        data = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y
        }

        export_stats(model=dnn, name=f"DNN_BATCH_{batch_list[i]}", params=params, num_runs=5, data=data, random=random_state)
    return 



def test_optimizers():

    learning_rate_adam = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    adam = {
        "name": "adam",
        "opt": keras.optimizers.Adam(
            learning_rate=learning_rate_adam,
            beta_1=beta_1,
            beta_2=beta_2
        ),
        "lr": learning_rate_adam
    }

    momentum = 0.9
    learning_rate_sgd = 0.01
    sgd = {
        "name": "sgd",
        "opt": keras.optimizers.SGD(
            learning_rate=learning_rate_sgd,   
            momentum=momentum
        ),
        "lr": learning_rate_sgd
    }

    learning_rate_adagrad = 0.001
    adagrad = {
        "name": "adagrad",
        "opt": keras.optimizers.Adagrad(
            learning_rate=learning_rate_adagrad
        ),
        "lr": learning_rate_adagrad
    }
    
    

    learning_rate_rms = 0.001
    rms = {
        "name": "rmsprop",
        "opt": keras.optimizers.RMSprop(
            learning_rate=learning_rate_rms,   
        ),
        "lr": learning_rate_rms
    }

    learning_rate_adadelta = 0.001
    adadelta = {
        "name": "adadelta",
        "opt": keras.optimizers.Adadelta(
            learning_rate=learning_rate_adadelta
        ),
        "lr": learning_rate_adadelta
    }

    
    """
    SGD
    Adam
    Adagrad
    rmsprop
    adadelta
    """
    optimizer_list = [sgd, adam, adagrad, rms, adadelta]
    data = pd.read_csv("train.csv")
    random_state = 1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random_state)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    for i in range(len(optimizer_list)):        
    
        cce = keras.losses.CategoricalCrossentropy()
    
        def scheduler(epoch, loss): 
            return loss
    
        l1_val = 0.01
        l1 = keras.regularizers.l1(l1_val)
    
        layers=[64]
        batch=1024
        epochs=200
        patience=15
        activation="relu"
        dnn = DNN(
            layers=layers,
            activation=activation,
            optimizer=optimizer_list[i]["opt"],
            loss=cce,
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.F1Score(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],
            batch=batch,
            epochs=epochs,
            early_stop=True,
            patience=patience,
            lrs=True,
            scheduler=scheduler,
            regularizer=l1
        )
        params = {
            "optimizer": optimizer_list[i]["name"],
            "learning rate": optimizer_list[i]["lr"],
            "beta_1": beta_1,
            "beta 2": beta_2,
            "loss": "categorical cross entropy",
            "lrs scheduler": "none",
            "regularizer": "l1",
            "l1": l1_val,
            "layers": layers,
            "batch": batch,
            "epochs": epochs,
            "patience": patience,
            "activation": activation
        }

        data = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y
        }

        export_stats(model=dnn, name=f"DNN_OPT_{optimizer_list[i]['name']}_LR_{optimizer_list[i]['lr']}", params=params, num_runs=5, data=data, random=random_state)
    return 



def test_regularizers():

    
    l1_val = 0.01
    l1 = {
        "name": "l1",
        "reg": keras.regularizers.l1(l1_val),
        "val": l1_val
    }

    l2_val = 0.01
    l2 = {
        "name": "l2",
        "reg": keras.regularizers.l2(l2_val),
        "val": l2_val
    }

    elastic = {
        "name": "elastic net",
        "reg": keras.regularizers.l1_l2(l1=l1_val, l2=l2_val),
        "val": [l1_val, l2_val]
    }

    # regularizer_list = [l1,l2,elastic]
    regularizer_list = [l2]
    data = pd.read_csv("train.csv")
    random_state = 1
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True, random=random_state)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    for i in range(len(regularizer_list)): 

        learning_rate = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        adam = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )   
    
        cce = keras.losses.CategoricalCrossentropy()
    
        def scheduler(epoch, loss): 
            # /2 every 20 epochs
            return loss/(2**math.floor(epoch/20))
        # def scheduler(epoch, loss): 
        #     return loss
    
        layers=[64]
        batch=1024
        epochs=200
        patience=15
        activation="relu"
        dnn = DNN(
            layers=layers,
            activation=activation,
            optimizer=adam,
            loss=cce,
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.F1Score(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],
            batch=batch,
            epochs=epochs,
            early_stop=True,
            patience=patience,
            lrs=True,
            scheduler=scheduler,
            regularizer=regularizer_list[i]["reg"]
        )
        params = {
            "optimizer": "adam",
            "learning rate": learning_rate,
            "beta_1": beta_1,
            "beta 2": beta_2,
            "loss": "categorical cross entropy",
            "lrs scheduler": "none",
            "regularizer": regularizer_list[i]["name"],
            "regularizer val": regularizer_list[i]["val"],
            "layers": layers,
            "batch": batch,
            "epochs": epochs,
            "patience": patience,
            "activation": activation
        }

        data = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y
        }

        export_stats(model=dnn, name=f"DNN_SCHED_.5_20epochs", params=params, num_runs=5, data=data, random=random_state)
    return 


def final_test():
    data = pd.read_csv("train.csv")
    random_state = 1
    train_x, train_y, val_x, val_y = preprocess_val(data, stdz=True, val_pct=.2, shuffle=True, random=random_state)

    test = pd.read_csv("test.csv")
    test = standardize(test)
    # test = test.to_numpy()

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)

    learning_rate = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    adam = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2
    )

    cce = keras.losses.CategoricalCrossentropy()

    def scheduler(epoch, loss): 
        return loss

    l2_val = 0.01
    l2 = keras.regularizers.l2(l2_val)

    layers=[64]
    batch=1024
    epochs=200
    patience=15
    dnn = DNN(
        layers=layers,
        activation="relu",
        optimizer=adam,
        loss=cce,
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.AUC(),
            keras.metrics.F1Score(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
        ],
        batch=batch,
        epochs=epochs,
        early_stop=True,
        patience=patience,
        lrs=True,
        scheduler=scheduler,
        regularizer=l2
    )
    params = {
        "optimizer": "adam",
        "learning rate": learning_rate,
        "beta_1": beta_1,
        "beta 2": beta_2,
        "loss": "categorical cross entropy",
        "lrs scheduler": "none",
        "regularizer": "l2",
        "l2": l2_val,
        "layers": layers,
        "batch": batch,
        "epochs": epochs,
        "patience": patience
    }

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test": test
    }

    export_stats(model=dnn, name=f"DNN_FINAL", params=params, num_runs=5, data=data, random=random_state, submission=True)
    return 



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
            "layers": [64],
            "batch_size": 1024,
            "activation": "relu",
            "epochs": 400,
            "patience": 15,
        }
    """

    layers = assign_param(params, "layers", i)
    activation = assign_param(params, "activation", i)
    optimizer = assign_params_nested(params, "optimizer", i)
    loss = assign_params_nested(params, "loss", i)
    batch = assign_param(params, "batch_size", i)
    epochs = assign_param(params, "epochs", i)
    patience = assign_param(params, "patience", i)
    scheduler = assign_params_nested(params, "scheduler", i)
    regularizer = assign_params_nested(params, "regularizer", i)

    dnn = DNN(
        layers=layers,
        activation=activation,
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

    return dnn


def assign_param(params: dict, key: str, i: int):
    """ 
    see build_model
    """
    if isinstance(params[key], list) and key != "layers":
        return params[key][i]
    else:
        return params[key]

def assign_params_nested(params: dict, key: str, i: int):
    """
    see build_model
    """
    if isinstance(params[key], list) and key != "layers":
        return copy.deepcopy(params[key][i][key])
    else:
        return copy.deepcopy(params[key][key])

def input_params_to_output_params(params: dict, i: int):
    out = copy.deepcopy(params)
    for k, v in out.items():
        if isinstance(out[k], list) and k != "layers":
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

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y
    }

    params = {
        "learning_rate": 0,
        "optimizer": [
            {
                "name": "adam 0.1",
                "optimizer": keras.optimizers.Adam(
                    learning_rate = 0.1,
                ),
            }, 
            {
                "name": "adam 0.01",
                "optimizer": keras.optimizers.Adam(
                    learning_rate = 0.01,
                ),
            }, 
            {
                "name": "adam 0.001",
                "optimizer": keras.optimizers.Adam(
                    learning_rate = 0.001,
                ),
            }
        ],
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
        "layers": [64],
        "batch_size": 1024,
        "activation": "relu",
        "epochs": 400,
        "patience": 15,
    }

    test_key = "optimizer"

    for i in range(len(params[test_key])):
        cnn = build_model(params, i)
        output_params = input_params_to_output_params(params, i)
        export_stats(model=cnn, name=f"DNN_{test_key}_{params[test_key][i]['name']}", params=output_params, num_runs=5, data=data, random=random, output_precision=5, plot=True)
    return


if __name__ == "__main__":
    test_param()



"""
wrapping up
run for LR = 0.1, 0.01, 0.001
for each model - best run of each LR
    plot loss, acc
    record acc, f1, auc

and best model AUC-ROC curve

"""

"""
testing list
- learning rate
    so far 0.9 might be the best?
    error on 0.001
    we can try 0.99 and 0.999
    0.9999 might be our best now? maybe we just got lucky with the rolls idk
    do you want to try higher lrs? or move on
    we can always come back
- layers
    i'm turning off the plots, just let it run
    [256] might be the best as prec/rec all go up
    we have to try multiple layers
    512, 256, 128 and stuff like that
    hmm, none if it is better than just [256]
    i can try with a normal LR
    i should try with softmax at the end...
    0.01 lasted longer for layers
    4096 had better recall (4096,2048,1024)
    that might be our best one but not significantly slow
    softmax was what i needed. consistently .85 accuracy, almost 1 AUC
    256 is still our best. we can test with lower nodes to see what happens
    and with two layers to see if that changes anything
    [64] is the best we have so far
    i want to try two layers, probably just doubles to see if that does anything
    [4 4] to [4096 4096]
    interrupted int he middle of 4096
    [16, 16] is the best, but [64] is still better
    i can try with changing the models, but we can do that later
- batches
    1024 is the best
- layers again, i don't want to think about optimizers rn
    just [64] is best
- optimizer
    SGD
    Adam
    Adagrad
    rmsprop
    adadelta
    Adam
    Adam is the best
- regularizer
    l1, l2, elastic
    l2 regularizer significantly better wow
- epochs
   not necessary
- patience
   also not necessary
- scheduler
   not as good, default loss is good enough
- optimizer parameters - kinda already optimized
- learning rate again?
  0.001 is good enough
"""


