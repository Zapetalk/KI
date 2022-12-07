import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import sklearn as sk
from sklearn.model_selection import train_test_split


img_height = 250
img_width = 250
batch_size = 1

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
train_label = np.hstack([y for x, y in ds_validation])


model = tf.keras.models.load_model('Gender_3.h5')
y_hat = model.predict(ds_validation)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(accuracy_score(train_label, y_hat))
