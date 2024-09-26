import logging

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pravljenje histograma za prikaz primera


Y = pd.read_csv("podaciB.csv", header=None)

plt.figure()
Y.hist(width=0.5, bins=8)
plt.show()

# Prikazivanje po jednog primera za svaku klasu

main_path = 'projekat_dataset/'
test_path = 'test/'

img_size = (64, 64)
batch_size = 64

from keras.utils import image_dataset_from_directory

N = 8

Xprimer = image_dataset_from_directory('primer/',
                                       image_size=img_size,
                                       shuffle=0)

classes = Xprimer.class_names
num_classes = len(classes)

print(classes)

plt.figure()
for img, lab in Xprimer.take(1):
    for i in range(N):
        plt.subplot(4, int(N / 4), i + 1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')

plt.show()

# Deljenje podataka na trening i validacioni skup


Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

Xtest = image_dataset_from_directory(test_path,
                                     image_size=img_size,
                                     batch_size=batch_size,
                                     seed=123)

# Predprocesiranje uz pomoc layers.Rescaling, layers.RandomFlip, layers.RandomRotation, layers.RandomZoom
# Obucavanje neuralne mreze koristeci dropout, L2 regularizaciju i rano zaustavljanje kao odbranu od preobucavanja

from keras import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.regularizers import l2

data_augmentation = Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
])

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_size[0], img_size[1], 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

es = EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)

history = model.fit(Xtrain,
                    epochs=100,
                    validation_data=Xval,
                    callbacks=[es],
                    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

from sklearn.metrics import accuracy_score

correct = {"cassava": True, "corn": True, "eggplant": True, "orange": True, "peperchili": True,
           "soybeans": True, "spinach": True, "watermelon": True}
incorrect = {"cassava": True, "corn": True, "eggplant": True, "orange": True, "peperchili": True,
             "soybeans": True, "spinach": True, "watermelon": True}
correct_img = []
correct_lab = []
incorrect_img = []
incorrect_lab = []
incorrect_pred = []

labels = np.array([])
pred = np.array([])

for img, lab in Xtest:
    prediction = np.argmax(model.predict(img, verbose=0), axis=1)
    labels = np.append(labels, lab)
    pred = np.append(pred, prediction)

    if not (any(correct) or any(incorrect)):
        continue
    for i in range(len(prediction)):
        if prediction[i] == lab[i] and correct[classes[lab[i]]]:
            correct[classes[lab[i]]] = False
            correct_img.append(img[i])
            correct_lab.append(lab[i])
        elif prediction[i] != lab[i] and incorrect[classes[lab[i]]]:
            incorrect[classes[lab[i]]] = False
            incorrect_img.append(img[i])
            incorrect_lab.append(lab[i])
            incorrect_pred.append(prediction[i])

print('Tačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')

plt.figure()
for i in range(N):
    plt.subplot(4, int(N / 4), i + 1)
    plt.imshow(correct_img[i].numpy().astype('uint8'))
    plt.title(f"{classes[correct_lab[i]]} correct")
    plt.axis('off')
plt.show()

plt.figure()
for i in range(N):
    plt.subplot(4, int(N / 4), i + 1)
    plt.imshow(incorrect_img[i].numpy().astype('uint8'))
    plt.title(f"{classes[incorrect_pred[i]]} instead of {classes[incorrect_lab[i]]}")
    plt.axis('off')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(10,10))
cmDisplay.plot(xticks_rotation='vertical',ax=ax)
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
    prediction = np.argmax(model.predict(img, verbose=0), axis=1)
    labels = np.append(labels, lab)
    pred = np.append(pred, prediction)

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(10,10))
cmDisplay.plot(xticks_rotation='vertical',ax=ax)
plt.show()
