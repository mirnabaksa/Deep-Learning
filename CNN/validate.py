from __future__ import division
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import model_from_json

img_width, img_height = 300, 300

json_file = open('model.json','r')
nn_json = json_file.read()
json_file.close()

weights_file = "model.hdf5"
model = model_from_json(nn_json)
model.load_weights(weights_file)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
			 
datagen = ImageDataGenerator(
        rescale=1./255)

test_generator = datagen.flow_from_directory(
		'./dataset/test', 
		batch_size=1,
		color_mode="rgb",
		class_mode="categorical",
      	shuffle=True,
		seed = 42,
		target_size=(img_width, img_height))
		
score = model.evaluate_generator(test_generator, steps=100)
print('test loss:', score[0])
print('test accuracy:', score[1])

test_generator.reset()
validation_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)

predictions = model.predict_generator(test_generator, steps=validation_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1) 

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

from sklearn.metrics import classification_report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
