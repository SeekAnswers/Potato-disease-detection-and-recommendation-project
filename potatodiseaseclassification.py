# %% [markdown]
# Importing Libraries for Deep Learning (TensorFlow), Building Neural Networks (Keras), Visualization (Matplotlib), and Warning Management (Warnings)
# 

# %%
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Defining/setting image processing parameters
# 

# %%
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS = 3

# %%
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\kccha\OneDrive\Desktop\Programming\Potato disease classification\dataset', shuffle=True, image_size = (IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)

# %%
#Lets's confirm the classes
classname = dataset.class_names
classname

# %%
#Let's check one of the images
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

# %% [markdown]
# Visualisation(let us see) of some of the images from the dataset<h>
# 

# %%
plt.figure(figsize=(10,10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(classname[labels_batch[i]])
        plt.axis('off')

# %% [markdown]
# <h>Next is splitting the dataset, we would separate into the version for training(70% to 80%), testing(10% to 20%) and validation(10%)
# 

# %%
#Let's check the length of the dataset
len(dataset)   #it is 32 because it is loading the images in the batch format

# %% [markdown]
# Let's divide the data now
# 

# %% [markdown]
# Let's separate the training data<h>
# 

# %%
train_size = 0.8 #training with 80% of dataset ie 80/100 = 0.8
len(dataset)*train_size

# %%
train_ds = dataset.take(54)
len(train_ds)

# %% [markdown]
# Next, let's separate the test data
# 

# %%
test_ds = dataset.skip(54)
len(test_ds)

# %%
val_size = 0.1
len(dataset) * val_size

# %%
val_ds = test_ds.take(6)
len(val_ds)

# %%
test_ds = test_ds.skip(6)
len(test_ds)

# %%
train_split=0.8
val_split=0.1
test_split=0.1

assert train_split+val_split+test_split == 1

# %%
def get_dataset_paratitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert train_split+val_split+test_split == 1

    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size) #val_size was left as it was same 0.1 as test_size

    return train_ds, val_ds, test_ds 

# %%
train_ds, val_ds, test_ds = get_dataset_paratitions_tf(dataset)

# %%
len(train_ds)

# %%
len(val_ds)

# %%
len(test_ds)

# %% [markdown]
# Let's put these in a Cache, Shuffle, and Prefetch the dataset
# 

# %%
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

# %% [markdown]
# Now on to building the model
# 

# %% [markdown]
# - Let's begin by data resizing and normalisation
# 

# %%
resize_and_rescale = tf.keras.Sequential([
    # layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    # layers.experimental.preprocessing.Rescaling(1.0/255) experimental.preprocessing has been removed from latest update to Keras
    layers.Rescaling(1.0/255)
])

# %% [markdown]
# - Then, Data Augmentation

# %%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2)
])

# %% [markdown]
# - Now on to applying to the trained data

# %%
train_ds = train_ds.map(
    lambda x,y : (data_augmentation(x, training=True), y)).prefetch(buffer_size = tf.data.AUTOTUNE)

# %% [markdown]
# - On to creating the model architecture

# %%
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),  # Added InputLayer to define input shape
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),  # Removed input_shape from here
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(n_classes, activation='softmax'), 
])

# No need to call model.build() again


# %% [markdown]
# - The model's ready, let's check a summary of it before compiling and printing

# %%
model.summary()

# %% [markdown]
# - Next is compling the model

# %%
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

# %% [markdown]
# Fitting the model

# %%
history = model.fit(
    train_ds,
    epochs = 10,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data = val_ds
)

# %% [markdown]
# Let's save the model

# %%
# model.save('potato_disease_model.h5') #Save as Legacy format
model.save('potato_disease_model.keras') #Save as native format as it will be upto date with latest Keras features

# %% [markdown]
# Let's evaluate the model

# %%
scores = model.evaluate(test_ds)

# %%
scores

# %%
history.params

# %%
history.history.keys()

# %%
type(history.history['loss'])

# %%
len(history.history['loss'])

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# %%
val_acc

# %% [markdown]
# Let's visualise the loss and accuracy using a graph

# %%
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(range(10), acc, label = 'Training Accuracy')
plt.plot(range(10), val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training & Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(10), loss, label = 'Training loss')
plt.plot(range(10), val_loss, label = 'Validation loss')
plt.legend(loc = 'upper right')
plt.title('Training & Validation loss')

# %% [markdown]
# Let's test with single images

# %%
import numpy as np
for image_batch , labels_batch in test_ds.take(1):
    first_img = image_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print('first image to predict')
    plt.imshow(first_img)
    print('actual label:', classname[first_label])

    batch_prediction = model.predict(image_batch)
    print('predicted label:', classname[np.argmax(batch_prediction[0])])

# %% [markdown]
# Creation of functions for prediction

# %%
def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)

    predicted_class = classname[np.argmax(prediction[0])]
    confidence = round(100 * (np.argmax(prediction[0])), 2)
    return predicted_class, confidence

# %%
plt.figure(figsize=(15, 15))

for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images[i])
        actual_class = classname[labels[i]]

        plt.title(f'Actual_class :{actual_class}, \n Predicted: {predicted_class}. \n Confidence: {confidence}%')

        plt.axis('off')



# %% [markdown]
# 


