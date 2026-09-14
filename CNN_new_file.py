import zipfile, os
local_zip = 'C:\Users\JULI YANDI RAHMAN\Downloads\Kerjaan\Project 2144\CNN_data.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

print("Jumlah Data Train Tiap Kelas")
print('Jumlah gambar gajah :', len(os.listdir('train/train/gajah/')))
print('Jumlah gambar kucing :', len(os.listdir('train/train/kucing/')))
print('Jumlah gambar burung :', len(os.listdir('train/train/burung/')))

os.mkdir('/tmp/s/')
os.mkdir('/tmp/s/train/')
os.mkdir('/tmp/s/val/')
os.mkdir('/tmp/s/train/gajah/')
os.mkdir('/tmp/s/val/gajah/')
os.mkdir('/tmp/s/train/kucing/')
os.mkdir('/tmp/s/val/kucing/')
os.mkdir('/tmp/s/train/burung/')
os.mkdir('/tmp/s/val/burung/')

import random
from shutil import copyfile

def train_val_split(source,train,val,train_ratio):
    total_size = len(os.listdir(source))
    train_size = int(train_ratio*total_size)
    val_size = total_size - train_size

    randomized = random.sample(os.listdir(source), total_size)
    train_files = randomized[0:train_size]
    val_files = randomized[train_size:total_size]

    for i in train_files:
        i_file = source + i
        destination = train + i
        copyfile(i_file,destination)

    for i in val_files:
        i_file = source + i
        destination = val + i
        copyfile(i_file,destination)
train_ratio = 0.8

#Pembagian Training dan Validasi

source_00 = 'train/train/gajah/'
train_00 = '/tmp/s/train/gajah/'
val_00 = '/tmp/s/val/gajah/'
train_val_split(source_00, train_00, val_00, train_ratio)

source_01 = 'train/train/kucing/'
train_01 = '/tmp/s/train/kucing/'
val_01 = '/tmp/s/val/kucing/'
train_val_split(source_01, train_01, val_01, train_ratio)

source_02 = 'train/train/burung/'
train_02 = '/tmp/s/train/burung/'
val_02 = '/tmp/s/val/burung/'
train_val_split(source_02, train_02, val_02, train_ratio)

print('Jumlah All gajah     :',len(os.listdir('train/train/gajah')))
print('Jumlah Train gajah     :',len(os.listdir('/tmp/s/train/gajah')))
print('Jumlah Val gajah     :',len(os.listdir('/tmp/s/val/gajah')))

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range = 30,
    horizontal_flip = True,
    shear_range = 0.3,
    fill_mode = 'nearest',
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.1
)

val_datagen = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range = 30,
    horizontal_flip = True,
    shear_range = 0.3,
    fill_mode = 'nearest',
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.1
)

train_dir = '/tmp/s/train/'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    batch_size = 3,
    class_mode = 'categorical'
)

val_dir = '/tmp/s/val/'
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (150, 150),
    batch_size = 3,
    class_mode = 'categorical'
)

#Callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print('\nAkurasi mencapai 99%')
            self.model.stop.training = True
callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation = 'relu'),
    tf.keras.layers.Dropout(0.3,seed=112),
    tf.keras.layers.Dense(500, activation = 'relu'),
    tf.keras.layers.Dropout(0.5, seed=112),
    tf.keras.layers.Dense(3, activation = 'softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=20,
    epochs=250,
    validation_data=val_generator,
    validation_steps=5,
    verbose=1,
    callbacks=[callbacks]
)

%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc = 'best')
plt.show()

import numpy as np
from keras.preprocessing import image
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    #Predicting Images
    path = fn
    img = image.load.img(path, target_size = (150, 150))
    imgplot = plt.imshow(img)
    x = image.img.to.array(img)
    x = np.expand_dims(x, axis = 0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size = 100)

    print(fn)

    class_list = os.listdir('train/train/')

    for j in range(42):
        if classes[0][j] == 1. :
            print('This image belongs to class', class_list[j])
            break