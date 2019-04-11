import pandas
from sklearn.metrics import confusion_matrix, classification_report
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.utils import np_utils
from collections import Counter
import argparse
import time


def review_to_words(raw_review):
	"""
	Only keeps ascii characters in the review
	"""
	try:
		letters_only = re.sub("[^a-zA-Z@]", " ", raw_review)
		words = letters_only.lower().split()
		words = list(filter(lambda word : word not in stopwords,words))
		return " ".join(words)
	except:
		return ""



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--id')
	args = parser.parse_args()

	id = args.id

	print("Starting:", time.ctime())
	stopwords = [line.rstrip('\n') for line in open('english')]
	
	############################################
	# Data
	reviews = pandas.read_csv("dataset.csv", index_col=0)
	reviews = reviews.sample(frac=1).reset_index(drop=True)
	

	# Pre-process the review and store in a separate column
	reviews['clean_review'] = reviews['reviewText'].apply(lambda x: review_to_words(x))
	
	# Join all the words in review to build a corpus
	all_text = ' '.join(reviews['clean_review'])
	words = all_text.split()

	# Convert words to integers
	counts = Counter(words)
	
	#numwords = 200  # Limit the number of words to use

	# Filter the words that have occured less than 200 times
	vocab = {x[0] : x[1] for x in counts.items() if x[1] >= 200}
	vocab = sorted(vocab.items(), key = lambda x:x[1], reverse=True)
	vocab = [i for (i,j) in vocab]
	vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
	numwords = len(vocab)

	review_ints = []
	for each in reviews['clean_review']:
		review_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])
	

	# Create a list of labels
	labels = np.array(reviews['sentiment'])

	# Find the number of reviews with zero length after the data pre-processing
	lens = [len(x) for x in review_ints]
	review_len = Counter(lens)
	print("Zero-length reviews: {}".format(review_len[0]))
	print("Maximum review length: {}".format(max(review_len)))

	# Remove those reviews with zero length and its corresponding label
	review_idx = [idx for idx, review in enumerate(review_ints) if len(review) > 0]
	labels = labels[review_idx]
	reviews = reviews.iloc[review_idx]
	review_ints = [review for review in review_ints if len(review) > 0]

	from math import ceil
	# Take the mean as the length delimiter - speeds up the training
	seq_len = ceil(np.mean(lens))
	print(seq_len)

	features = np.zeros((len(review_ints), seq_len), dtype=int)
	for i, row in enumerate(review_ints):
		features[i, -len(row):] = np.array(row)[:seq_len]

	split_frac = 0.8
	split_idx = int(len(features) * 0.8)
	train_x, val_x = features[:split_idx], features[split_idx:]
	train_y, val_y = labels[:split_idx], labels[split_idx:]

	test_idx = int(len(val_x) * 0.5)
	val_x, test_x = val_x[:test_idx], val_x[test_idx:]
	val_y, test_y = val_y[:test_idx], val_y[test_idx:]

	print("\t\t\tFeature Shapes:")
	print("Train set: \t\t{}".format(train_x.shape),
		  "\nValidation set: \t{}".format(val_x.shape),
		  "\nTest set: \t\t{}".format(test_x.shape))

	print("Train set: \t\t{}".format(train_y.shape),
		  "\nValidation set: \t{}".format(val_y.shape),
		  "\nTest set: \t\t{}".format(test_y.shape))

	############################################
	# Model
	
	nlayers = 1
	#nlayers = 2 
	#nlayers = 3
	
	#neurons = 128
	neurons = 256
	#neurons = 512

	drop = 0.0
	#drop = 0.1
	#drop = 0.3
	#drop = 0.5
	#drop = 0.7

	embedding = 20
	RNN = LSTM  # GRU
	
	verbose = 1
	impl = 2
	numclasses = 3

	model = Sequential()
	model.add(Embedding(numwords+1, embedding, input_length=seq_len))

	if nlayers == 1:
		model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop))
	else:
		model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, return_sequences=True))
		for i in range(1, nlayers - 1):
			model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))
		model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

	model.add(Dense(numclasses))
	model.add(Activation('softmax'))

	############################################
	# Training

	#learning_rate = 0.001
	learning_rate = 0.01
	#learning_rate = 0.1
	
	# Default Keras values are used for optimizer params
	momentum = 0.95
	#momentum = 0
	#momentum = 0.5
	#momentum = 1
	
	optimizer = SGD(lr=learning_rate, momentum=momentum)
	#optimizer = Adadelta(lr=1.0)
	#optimizer = RMSprop(lr=0.001)
	#optimizer = Adagrad(lr=0.01)
	#optimizer = Adam(lr=0.001)
	#optimizer = Adamax(lr=0.002)
	#optimizer = Nadam(lr=0.002)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	epochs = 50
	batch_size = 64

	train_y_c = np_utils.to_categorical(train_y, numclasses)
	val_y_c = np_utils.to_categorical(val_y,numclasses)

	history = model.fit(train_x, train_y_c,
			  batch_size=batch_size,
			  epochs=epochs,
			  validation_data=(val_x, val_y_c), verbose = verbose)

	############################################
	# Results

	test_y_c = np_utils.to_categorical(test_y, numclasses)
	score, acc = model.evaluate(test_x, test_y_c, batch_size=batch_size)
					
	print()
	print('Test ACC=', acc)

	test_pred = model.predict_classes(test_x)

	# Calculate metrics
	print()
	print('Confusion Matrix')
	print('-'*20)
	print(confusion_matrix(test_y, test_pred))
	print()
	print('Classification Report')
	print('-'*40)
	print(classification_report(test_y, test_pred))
	print()
	print("Ending:", time.ctime())


	
	# Plot results
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
	plt.savefig(str(id) + 'accuracy.pdf')
	plt.close()

	# Loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','validation'], loc='upper left')
	plt.savefig(str(id) + 'loss.pdf')

