# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesf = train_images / 255.0

test_imagesf = test_images / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_imagesf, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_imagesf,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_imagesf.reshape(-1, 28, 28))
pred_20 = predictions[20]
print(pred_20)
max_20 = np.argmax(pred_20)
print(max_20)
