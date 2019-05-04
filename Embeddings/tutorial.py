import re
import multiprocessing
import gensim
import numpy as np
import logging  

from time import time 
from gensim.models import Word2Vec

def print_most_similar(word):
	print(word)
	print_formatted_output(w2v_model.wv.most_similar(positive=[word]))
	print("\n")

def print_formatted_output(output):
	print(''.join(x[0] + "  " + '{0:.4f}'.format(x[1]) + "\n" for x in output))

def cleaning(raw_text):	
	letters_only = re.sub("[^a-zA-Z@]", " ", raw_text)
	words = letters_only.lower().split()
	words = list(filter(lambda word : word not in stopwords,words))
	if len(words) == 0:
		return None
	return words
		
if __name__ == '__main__':
	logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
	
	stopwords = [line.rstrip('\n') for line in open('english')]

	with open("books/HP1.txt",errors='ignore') as f:
		text = f.read()
	with open("books/HP2.txt",errors='ignore') as f:
		text += f.read()
	with open("books/HP3.txt",errors='ignore') as f:
		text += f.read()
	with open("books/HP4.txt",errors='ignore') as f:
		text += f.read()
	with open("books/HP5.txt",errors='ignore') as f:
		text += f.read()
	with open("books/HP6.txt",errors='ignore') as f:
		text += f.read()
	with open("books/HP7.txt",errors='ignore') as f:
		text += f.read()
	
	sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
	t = time()
	
	sentences = [cleaning(sentence) for sentence in sentences]
	sentences = [x for x in sentences if x is not None]
	
	print('Clean up took: {} minutes'.format(round((time() - t) / 60, 3)))

	cores = multiprocessing.cpu_count()
	model = Word2Vec(min_count=10,
                     window=3,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
	t = time()
	model.build_vocab(sentences, progress_per=10000)

	
	print('Vocab building took: {} minutes'.format(round((time() - t) / 60, 3)))

	t = time()

	model.train(sentences, 
		total_examples=model.corpus_count, 
		epochs=50, 
		report_delay=1)
		
	print('Training took: {} minutes'.format(round((time() - t) / 60, 3)))
	model.save("word2vec.model")
		