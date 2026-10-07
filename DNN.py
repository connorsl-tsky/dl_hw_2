
import keras
import pandas as pd
from Preprocessing import preprocess_test_val
import numpy as np
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


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
        self.model.add(keras.layers.Dense(units=10)) # 10 for the digits
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
        
    


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True)

    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    learning_rate = 0.1
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

    l1 = keras.regularizers.l1(0.01)

    layers = [128]
    batch=128
    epochs=100
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
        regularizer=l1
    )

    loss = dnn.train(train_x, train_y, val_x, val_y)
    print("LOSS", loss.history)
    # l, acc, auc, f1, prec, rec = dnn.evaluate(test_x, test_y)
    # print("L", l)
    # print("ACC", acc)
    # print("AUC", auc)
    # print("f1", f1)
    # print("PREC", prec)
    # print("REC", rec)

    yhat = np.array(dnn.model.predict(test_x))
    print("YHAT", yhat)
    print("YHAT shape", yhat.shape)

    display = RocCurveDisplay.from_predictions(
            test_y.ravel(),
            yhat.ravel(),
            name="micro-average OvR",
            curve_kwargs=dict(color="darkorange"),
            plot_chance_level=True,
            despine=True,
        )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Title",
    )
    plt.show()
    # got it


    """
    auc-roc curve
    from sklearn.metrics import RocCurveDisplay
    score is result from predictions
    onehot_test is ytest iwth the to_classifier as one-hot encoding

    display = RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        curve_kwargs=dict(color="darkorange"),
        plot_chance_level=True,
        despine=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
    )
    
    """