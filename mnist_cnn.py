# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras import optimizers

"""
Name - Ghalib Saleem 



********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************

Results I got by adding CNN layers:-(3 Execution)

Epoch 6/6
235/235 [==============================] - 17s 72ms/step - loss: 0.0227 - accuracy: 0.9926 - val_loss: 0.0486 - val_accuracy: 0.9860

79/79 [==============================] - 3s 37ms/step - loss: 0.0435 - accuracy: 0.9875
test loss, test acc: [0.043459199368953705, 0.987500011920929]

--------------------------------------------------------------------------------

Epoch 6/6
235/235 [==============================] - 17s 71ms/step - loss: 0.0180 - accuracy: 0.9946 - val_loss: 0.0480 - val_accuracy: 0.9847

79/79 [==============================] - 3s 38ms/step - loss: 0.0415 - accuracy: 0.9862
test loss, test acc: [0.04150567576289177, 0.9861999750137329]

--------------------------------------------------------------------------------


Epoch 6/6
235/235 [==============================] - 17s 71ms/step - loss: 0.0180 - accuracy: 0.9939 - val_loss: 0.0557 - val_accuracy: 0.9850

79/79 [==============================] - 3s 38ms/step - loss: 0.0375 - accuracy: 0.9870
test loss, test acc: [0.03754426911473274, 0.9869999885559082]


********************************************************************************
********************************************************************************
********************************************************************************

Explainantion: I started by just adding only one convolution layer with minimum possible filters but the result was not that promising so I increased the filter count and added same padding. Also added another layer to get a better accuracy.
The reason I choose Max pooling over average pooling is because max pooling extracts the most impotant features like edges on the other hand average pooling extracts feature with smoothing and the dataset we are working on require feature extraction with edges. 
So this combination of convolution layers and pooling layer will focus on handwritten letters which is just a form of edges.
"""

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

(ds_train, dsvalid, ds_test), ds_info = tfds.load(
    'mnist',
# First 25% and last 25% from training, then validation data is 5%
# from 25% of train data to 30% and test is the usual 10K
    split=['train[:25%]+train[-25%:]','train[25%:30%]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

dsvalid = dsvalid.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dsvalid = dsvalid.batch(64)
dsvalid = dsvalid.cache()
dsvalid = dsvalid.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',activation ='relu',input_shape = (28,28,1)),
  tf.keras.layers.MaxPool2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'),
  #tf.keras.layers.MaxPool2D(pool_size=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    ds_train,
    epochs=6,
#    validation_data=ds_test,
    validation_data=dsvalid,
)

results = model.evaluate(ds_test, batch_size=128)
print("test loss, test acc:", results)

