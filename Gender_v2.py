from datetime import datetime
import tensorflow as tf



img_height = 64
img_width = 64
batch_size = 10

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


num_output_classes = 2
input_img_size = (64, 64, 1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_output_classes, activation="softmax"))
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)

log_dir = "logs/Gender/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#model.fit(ds_train, epochs=30, steps_per_epoch=70, validation_data=ds_validation, validation_steps=50,callbacks=[tensorboard_callback])
model.fit(ds_validation,
          batch_size=batch_size,
          epochs=100,
          verbose=1,
          callbacks=[tensorboard_callback])
model.save('Gender_v2_1.h5')


