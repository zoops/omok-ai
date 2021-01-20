import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

w, h = 20, 20
base_path = os.path.join('dataset', '*/*.npz')

file_list = glob(base_path)

x_data, y_data = [], []
for file_path in tqdm(file_list):
    data = np.load(file_path)
    x_data.extend(data['inputs'])
    y_data.extend(data['outputs'])

x_data = np.array(x_data, np.float32).reshape((-1, h, w, 1))
y_data = np.array(y_data, np.float32).reshape((-1, h * w))

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2020)

del x_data, y_data

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

model = models.Sequential([
    layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=(h, w, 1)),
    layers.Conv2D(128, 7, activation='relu', padding='same'),
    layers.Conv2D(256, 7, activation='relu', padding='same'),
    layers.Conv2D(128, 7, activation='relu', padding='same'),
    layers.Conv2D(64, 7, activation='relu', padding='same'),
    layers.Conv2D(1, 1, activation=None, padding='same'),
    layers.Reshape((h * w,)),
    layers.Activation('sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.summary()

start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('models', exist_ok=True)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=256,
    epochs=10,
    callbacks=[
        ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto')
    ],
    validation_data=(x_val, y_val),
    use_multiprocessing=True,
    workers=16
)

i = 0

for y in range(h):
    for x in range(w):
        print('%2d' % x_val[i][y, x], end='')
    print()


y_pred = model.predict(np.expand_dims(x_val[i], axis=0)).squeeze()
y_pred = y_pred.reshape((h, w))

y, x = np.unravel_index(np.argmax(y_pred), y_pred.shape)

print(x, y, y_pred[y, x])


