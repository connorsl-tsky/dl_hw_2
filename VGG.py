"""
https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

yeah just follow this guide. it's a special kind of CNN
"""

import keras
import pandas as pd
from Preprocessing import preprocess_test_val, reshape_for_conv
import numpy as np
import tensorflow as tf

class VGG:
    def __init__(self, optimizer: keras.Optimizer, loss: keras.Loss, metrics: list[keras.Metric], batch: int, epochs: int, early_stop: bool, patience: int | None, lrs: bool, scheduler):
        # member vars
        self.batch = batch
        self.epochs = epochs

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
        self.model.add(keras.layers.Dense(units=4096,activation='relu',))
        self.model.add(keras.layers.Dense(units=4096,activation='relu',))
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
        
    


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    print("PREPROCESSING")
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess_test_val(data, stdz=True, test_pct=.2, val_pct=.2, shuffle=True)

    print("TO CATEGORICAL")
    train_y = keras.utils.to_categorical(train_y, 10)
    val_y = keras.utils.to_categorical(val_y, 10)
    test_y = keras.utils.to_categorical(test_y, 10)

    print("RESHAPE FOR CONV")
    train_x = reshape_for_conv(train_x)
    val_x = reshape_for_conv(val_x)
    test_x = reshape_for_conv(test_x)

    print("RESIZE IMAGE")
    train_x = tf.image.resize(train_x, (32, 32)).numpy()
    val_x   = tf.image.resize(val_x,   (32, 32)).numpy()
    test_x  = tf.image.resize(test_x,  (32, 32)).numpy()


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

    batch=128
    epochs=100
    patience=15
    vgg = VGG(
        optimizer=adam,
        loss=cce,
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
    )

    """
    okay i'm getting an issue with this what the hell
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    ah with each max, the input dimensions divide by 2
    we have 5 maxes
    it's 28
    14,7,3,1 and boom, not enough. 
    hm. 
    whta's the minimum it needs to be
    1,2,4,8,16,32
    # 32
    # i'm doing research, it's not working, i'm gonna ask ai fuck it
    # """
    # print("train_x", train_x.shape)
    # conv1 = keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(train_x)
    # print("conv1", conv1.shape)
    # max1 = keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(conv1)
    # print("max1", max1.shape)
    # conv2 = keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(max1)
    # print("conv2", conv2.shape)
    # max2 = keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(conv2)
    # print("max2", max2.shape)
    # max3 = keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(max2)
    # print("max2", max3.shape)


    

    print("train x", train_x.shape)
    print("train y", train_y.shape)
    print("val x", val_x.shape)
    print("val y", val_y.shape)
    loss = vgg.train(train_x, train_y, val_x, val_y)
    print("LOSS", loss.history)
    l, acc, auc, f1, prec, rec = vgg.evaluate(test_x, test_y)
    print("L", l)
    print("ACC", acc)
    print("AUC", auc)
    print("f1", f1)
    print("PREC", prec)
    print("REC", rec)
    # print("IOU", iou)
    