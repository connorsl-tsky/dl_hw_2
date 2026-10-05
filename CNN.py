"""
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

"""


import keras
import pandas as pd
from Preprocessing import preprocess_test_val, reshape_for_conv

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

    train_x = reshape_for_conv(train_x)
    val_x = reshape_for_conv(val_x)
    test_x = reshape_for_conv(test_x)

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

    filters=32
    kernel_size=(3,3)
    strides=(1,1)
    padding="same"
    pool_size=(3,3)
    dense_units=128
    batch=128
    epochs=100
    cnn = CNN(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation="relu",
        pool_size=pool_size,
        dense_units=dense_units,
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
        patience=15,
        lrs=True,
        scheduler=scheduler,
        regularizer=l1
    )

    print("train x", train_x.shape)
    print("train y", train_y.shape)
    print("val x", val_x.shape)
    print("val y", val_y.shape)
    loss = cnn.train(train_x, train_y, val_x, val_y)
    print("LOSS", loss.history)
    l, acc, auc, f1, prec, rec = cnn.evaluate(test_x, test_y)
    print("L", l)
    print("ACC", acc)
    print("AUC", auc)
    print("f1", f1)
    print("PREC", prec)
    print("REC", rec)
    # print("IOU", iou)
    