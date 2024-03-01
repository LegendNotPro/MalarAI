```py
#imports
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import os
import PIL
from PIL import Image
from glob import glob
import json
from flask import url_for
#load_dataset
dataset, dataset_info = tfds.load("malaria",with_info=True,as_supervised=True,shuffle_files=True,split=['train'])
#split dataset
def splits(dataset,TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

    test_val_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
    val_dataset = test_val_dataset.take(int(VAL_RATIO*DATASET_SIZE))

    test_dataset = test_val_dataset.skip(int(TEST_RATIO*DATASET_SIZE))

    return train_dataset, test_dataset, val_dataset

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

train_dataset, test_dataset, val_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
#data preprocessing
IM_SIZE = 224

def resize_rescale(image, label):
  return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#creating model
def create_model():
  model = tf.keras.Sequential([
                              InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),

                              Conv2D(filters=6, kernel_size=3, strides=1, padding="valid", activation="relu"),
                              BatchNormalization(),
                              MaxPool2D(pool_size=2, strides=2),

                              Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", activation="relu"),
                              BatchNormalization(),
                              MaxPool2D(pool_size=2, strides=2),

                              Flatten(),

                              Dense(100, activation="relu"),
                              BatchNormalization(),
                              Dense(10, activation="relu"),
                              BatchNormalization(),
                              Dense(1, activation="sigmoid")

  ])
  model.compile(optimizer=Adam(learning_rate=0.01),
              loss=BinaryCrossentropy(),
              metrics="accuracy")
  return model
#training model


model = create_model()

model.summary()
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(train_dataset,validation_data=val_dataset, epochs=20, verbose=1, callbacks=[early_stopping])

model.save("malariav2.keras")
#prediction
def parasite_or_not(x):
  if x<0.5:
    return str("Infected")
  else:
    return str("Uninfected")

parasite_or_not(model.predict(test_dataset.take(1))[0][0])
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
def load_image():
  dir = "/content/drive/MyDrive/static/malaria_samples"
  glob_dir = "/content/drive/MyDrive/static/malaria_samples/*.png"
  onlyfiles = []

  for root, dirs, files in os.walk(dir):
    onlyfiles.extend(files)

  if len(onlyfiles) > 0:
    files = glob(glob_dir)
    for f in files:
      os.remove(f)

  #reshuffle
  reshuffled = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  for i, (image,label) in enumerate(test_dataset.take(4)):
    img = tensor_to_image(image)
    url = f"/content/drive/MyDrive/static/malaria_samples/sample_img{i}.png"
    img.save(url)

    final_label = str(parasite_or_not(label.numpy()[0]))
    final_prediction = str(parasite_or_not(model.predict(image)[0][0]))
    yield [f"malaria_samples/sample_img{i}.png", final_label, final_prediction]
def get_img_array():
  img_arr = []
  for i in load_image():
    img_arr.append([i[0], i[1], i[2]])
  return img_arr
from flask import Flask, render_template, jsonify, url_for

static_dir = os.path.abspath("/content/drive/MyDrive/static")
template_dir = os.path.abspath("/content/drive/MyDrive")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir,  static_url_path='/static')


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict')
def predict():
  pred_array = get_img_array()

  url1 = pred_array[0][0]
  url2 = pred_array[1][0]
  url3 = pred_array[2][0]
  url4 = pred_array[3][0]

  pred_array[0][0] = url_for("static", filename=url1)
  pred_array[1][0] = url_for("static", filename=url2)
  pred_array[2][0] = url_for("static", filename=url3)
  pred_array[3][0] = url_for("static", filename=url4)

  pred_dict = {"data": pred_array}
  print("Response data:", pred_dict)
  return jsonify(pred_dict)

app.run()
```
