"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

// Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

// Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

// One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(32, 32, 3))

// Freeze the base model
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')  
])

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x_train, y_train, 
          batch_size=64, 
          epochs=10, 
          validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

"""


"""
https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

yeah just follow this guide. it's a special kind of CNN
"""

import keras
import pandas as pd
from Preprocessing import preprocess_test_val, reshape_for_conv
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50


class ResNet:
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
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

        self.base_model.trainable = False
        self.model = keras.models.Sequential()
        self.init_layers()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return

    def init_layers(self):
        self.model.add(keras.layers.Conv2D(filters=3,kernel_size=(1,1),activation='relu'))
        self.model.add(self.base_model)
        self.model.add(keras.layers.GlobalAveragePooling2D())
        self.model.add(keras.layers.Dense(1024, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
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

    """
    argh
    resnet requires 3 channels
    is that becuase it's pretrained?
    is it possible to resize the inputs to 3 channels?
    does input layer do anything
    nah 
    
    ah pretraining
    add a conv layer before
    adds probably several seconds of trianing but oh well
    433pm
    433pm takes about a minute
    40 seconds
    okay. better
    """

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
    ResNet = ResNet(
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
    
    print("train x", train_x.shape)
    print("train y", train_y.shape)
    print("val x", val_x.shape)
    print("val y", val_y.shape)
    loss = ResNet.train(train_x, train_y, val_x, val_y)
    print("LOSS", loss.history)
    l, acc, auc, f1, prec, rec = ResNet.evaluate(test_x, test_y)
    print("L", l)
    print("ACC", acc)
    print("AUC", auc)
    print("f1", f1)
    print("PREC", prec)
    print("REC", rec)
    # print("IOU", iou)
    