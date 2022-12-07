import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import regularizers, models
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import layers

img_height = 250
img_width = 250
batch_size = 1
kernel_s=(1,1)


model=models.Sequential()
model.add(layers.Conv2D(32,input_shape=(250,250,1),
                        kernel_regularizer=regularizers.l2(0.001),kernel_size=4,padding="VALID"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,kernel_s))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,kernel_s,activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'GenderDataset/Training',
    labels='inferred',
    label_mode="int",
    class_names=['female', 'male'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshapes bc different heights and width
    shuffle=True,
)
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'GenderDataset/Validation',
    labels='inferred',
    label_mode="int",
    class_names=['female', 'male'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshapes bc different heights and width
    shuffle=True,
)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
log_dir = "logs/Gender/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(ds_train, epochs=100, steps_per_epoch=100, validation_data=ds_validation, validation_steps=50,
          callbacks=[tensorboard_callback])
model.save('Gender_4.h5')
train_label = np.hstack([y for x, y in ds_validation])
#Get Accrusy
y_hat = model.predict(ds_validation)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(accuracy_score(train_label, y_hat))

