import os
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import accuracy_score

img_height = 100
img_width = 80
batch_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Input((100, 80, 1)),
    tf.keras.layers.Conv2D(16, 3, padding='same'),
    tf.keras.layers.Conv2D(16, 3, padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

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
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True)],
    metrics=["accuracy"]
)
log_dir = "logs/Gender/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(ds_train, epochs=10, verbose=2, callbacks=[tensorboard_callback])
model.save('Gender')
#Get Accrusy
y_hat = model.predict(ds_validation)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(accuracy_score(ds_validation))

