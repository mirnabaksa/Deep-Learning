from __future__ import division
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import time

start = time.time()
img_width, img_height = 300, 300

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

batch_size = 32
epochs = 100

train_datagen = ImageDataGenerator(rescale=1./255)		
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './dataset/train', 
		color_mode="rgb",
	    	batch_size=batch_size,
		class_mode="categorical",
		shuffle=True,
		seed = 42,
        	target_size=(img_width, img_height))

val_generator = val_datagen.flow_from_directory(
        './dataset/validation', 
		color_mode="rgb",
		class_mode="categorical",
       	 	batch_size=batch_size,
		shuffle=True,
		seed = 42,
        	target_size=(img_width, img_height))

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(64, 3, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(47, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = val_generator.n//val_generator.batch_size

history = model.fit_generator(
        train_generator,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=train_generator.n/batch_size,
	validation_steps=val_generator.n/batch_size,
	validation_data=val_generator)		
		
score = model.evaluate_generator(val_generator, steps=1)
print('validation loss:', score[0])
print('validation accuracy:', score[1])

end = time.time()
duration = end-start

if duration<60:
    print("Experiment Duration:",duration,"seconds")
elif duration>60 and duration<3600:
    duration=duration/60
    print("Experiment Duration:",duration,"minutes")
else:
    duration=duration/(60*60)
    print("Experiment Duration:",duration,"hours")
				
from keras.models import model_from_json

model_json = model.to_json()
with open('model1.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "model1.hdf5"
model.save_weights(weights_file,overwrite=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('accuracy_1.pdf')
plt.close()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('loss_1.pdf')
