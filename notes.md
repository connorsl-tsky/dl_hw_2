# code / commands
 - create venv - python -m venv ./.venv
 - see if there are any NaNs in dataframe - train.isnull().any(), .any().any(), "True" in train.isnull().any().to_string()
 - get versions of packages - pip freeze and perhaps grep

# models


## dev

 - how to use tfdata sets
   - fark this we use sklearn
   - good choice

## DNN
https://www.geeksforgeeks.org/deep-learning/implementing-neural-networks-using-tensorflow/
feed forward neural network

// preprocessing
input_shape = [X_train.shape[1]]
model = tf.keras.Sequential([

    tf.keras.layers.Dense(units=64, activation='relu',
                          input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.summary()

model.compile(optimizer='adam', loss='mae') 

losses = model.fit(X_train, y_train,

                   validation_data=(X_val, y_val),
                   batch_size=256, 
                   epochs=15, 
  )
model.predict(X_val.iloc[0:3, :])
loss_df = pd.DataFrame(losses.history)
loss_df.loc[:,['loss','val_loss']].plot() # ah validation loss

## ConvNet

 - what is convnet?
   - oh it's just a cnn

- https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/
  - input layers - width, height, depth
  - convolutional layer - extract important features via kernels
  - activation layer - introduce nonlinearity
  - pooling layer - reduce dimensions to make training faster, reduce memory, prevent overfitting, typically between conv layers
  - flattening - multi dimensional to 1d vector to be passed to fully connected layer for classification
  - fully connected - produces final classification scores
  - output - final scores into probabilities

laplacian kernel for edge detection / important features
  kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=tf.float32)

kernel = tf.reshape(kernel, [3, 3, 1, 1])

conv_output = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],
    padding='SAME'
)

relu_output = tf.nn.relu(conv_output)

pool_output = tf.nn.max_pool2d(
    input=relu_output,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME'
)

flatten_layer = tf.keras.layers.Flatten()
flatten_output = flatten_layer(pool_output)

dense_layer = tf.keras.layers.Dense(
    units=64,         
    activation='relu' 
)

dense_output = dense_layer(flatten_output)

https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-tensorflow/

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


## VGG

https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

yeah just follow this guide. it's a special kind of CNN

## ResNet18

https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/

it's in pytorch so you might be able to figure out the architecture, but it's just a fancy convnet

https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/

some code

is there a built in resnet in keras?

https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
yeah, i'm not sure how good it is but whatever

https://www.geeksforgeeks.org/computer-vision/image-classification-using-resnet/
this is probably better

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

## determine possible layers, optimizers, the loss function, how to split the data, metrics, anything else? early stop, activation functions, regularization

 - layers
   - https://keras.io/api/layers/
   - dense, conv2d, maxpooling2d(?), batchnorm and other types of norm, dropout, flatten
 - optimizers
   - https://keras.io/api/optimizers/
   - SGD, Adam, Adagrad, rmsprop, adadelta, adam (preferred)
 - loss functions
   - cross entropy - probably categorical cross entropy
 - metrics
   - accuracy, AUC, recall, precision, f1, IoU
 - early stop
   - https://www.geeksforgeeks.org/deep-learning/using-early-stopping-to-reduce-overfitting-in-neural-networks/
 - activation functions
   - just use relu
 - regularization
   - l1, l2, l1l2, max-norm, orthogonal
 - momentum
 - learning rate scheduler
   - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
   - function that takes in epocha dn lr and returns the learning rate. perhaps something to test
 - split layers
   - https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset
   - data = np.random.random(size=(1000, 4))
   - left_ds, right_ds = keras.utils.split_dataset(data, left_size=0.8)
   - update: use sklearn
   



# data
wikipedia on mnist - https://en.wikipedia.org/wiki/MNIST_database

 - *what is data augmentation - from the wikipedia
 - *what is keras utils to_categorical?
    - https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
      - converts vector class to matrix so if it's a 1 it'll be the first column is a 1 and a 2 the second column is a 2 and the rest 0's. that kind of thing. not sure why that would be helpful tbh. 
 - *what is affine transformation? is there a way to do that with keras?
   - https://machinelearningplus.com/linear-algebra/affine-transformation/
     - preserver collinearity and ratios of distances - change in shape / rotation / position but basic structure is preserved
     - i assume this is just a category of transformation
 - *what is elastic distortion? is there a way to do that with keras?
   - https://schneppat.com/elastic-transformations.html
     - non linear distortions in data. probably in counter to affine transformations
 - *is there a way to do rotation and translation with keras?
   - yeah i think see below about data augmentation


 - describe the data
   - it seems pretty simple. we have 28^2 784 pixels in a 28x28 image, and the number is an integer that is 0 to 255 that indicates brightness or darkness. i assume that means all rgb are that number. and the first column is the label, 0 through 9
   - some columns are more filed out than others, like the first row is usually blank. i'm not sure if that means anything. 
   - but i know we don't really have to worry about skew or missing labels or anything. i'm pretty sure. 
   - that wouldn't be bad to check if there is any null
   - and test.csv just doesn't have any labels
   - there are no null values
   - i could divide everything by 255 to get everything down to a float or something. or between 0 and 1 for the uh standardization
 - preprocessing from notes
   - zero padding - probably not necessary since their might also be zero padding?
   - not it's not all around, the first row doesn't necessarily have padding, so that's an option
   - data augmentation - augment data to make it less sensitive to noise e.g. rotations, cropping, flipping, adding noise, masking regions of images, 
   - i think that's all from my notes
 - preprocessing from research online
   - https://www.geeksforgeeks.org/machine-learning/mnist-dataset/
     - it's already preprocessed
     - oh yeah they divide by 255 - data normalization
   - https://www.oreilly.com/library/view/machine-learning-for/9781789536300/ecbcd648-4db0-4ff2-aeff-383032b59832.xhtml
     - not a full source, but maybe randomly seed the data for better consistency and comparison when testing
   - https://deepwiki.com/li-haojia/pytorch-beginner-mnist/3.1-data-loading-and-preprocessing
     - use built in standardizer with mean and stdev
     - contains notes in other tabs on this persons models and stuff that might be useful for figuring out how to structure my CNN. and other models. i think this one is just CNN
   - https://www.geeksforgeeks.org/deep-learning/applying-convolutional-neural-network-on-mnist-dataset/
     - building a cnn, might be useful
     - but for preprocessing, what is keras utils to_categorical?
   - summary
     - generally, i don't think anything is necessary for this
     - besides either standardizing by a standardizer (which idek if keras has one, probably) or by dividing everything by 255. we can determine from there
     - a lot of the ones on wikipedia use affine transformation, elastic distortion, both, or  rotation and translation. but no affine for CNN. 
  - data augmentation - https://www.datacamp.com/tutorial/complete-guide-data-augmentation
     - modified versions of existing data to increase training size and resist overfitting
     - images - transformations (but not mulitple on same image), blurring (kernel filters), random erasing, mixing images
     - random flip, random rotation layers, 
     - has a bunch of tensorflow examples, but i'm not sure if any of them would be useful for what we're doing since the test set is only slightly different from the training set, yeah? could be worth a shot though, it doesn't look too difficult to implement