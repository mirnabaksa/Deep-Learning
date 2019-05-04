import gensim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
	

def plot(model, word, additional_words = None):
	sns.set_style("darkgrid")

	arrays = np.empty((0, 300), dtype='f')
	labels = [word]
	colors  = ['blue']

	arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
	similar = model.wv.most_similar([word], topn=10)
	
	for wrd_score in similar:
		wrd_vector = model.wv.__getitem__([wrd_score[0]])
		labels.append(wrd_score[0])
		colors.append('green')
		arrays = np.append(arrays, wrd_vector, axis=0)
	
	if additional_words is not None:
	
		for wrd in additional_words:
			wrd_vector = model.wv.__getitem__([wrd])
			labels.append(wrd)
			colors.append('red')
			arrays = np.append(arrays, wrd_vector, axis=0)
	   
	reduced = PCA(n_components=11).fit_transform(arrays)
	np.set_printoptions(suppress=True)
	
	Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduced)
	
	dataset = pd.DataFrame({'x': [x for x in Y[:, 0]],
					   'y': [y for y in Y[:, 1]],
					   'words': labels,
					   'color': colors})
	
	fig, _ = plt.subplots()
	fig.set_size_inches(6, 6)
	
	plot = sns.regplot(data=dataset,
					 x="x",
					 y="y",
					 fit_reg=False,
					 marker="o",
					 scatter_kws={'s': 40,
								  'facecolors': dataset['color']
								 }
					)
	
	for line in range(0, dataset.shape[0]):
		 plot.text(dataset["x"][line],
				 dataset['y'][line],
				 '  ' + dataset["words"][line].title(),
				 horizontalalignment='left',
				 verticalalignment='bottom', size='medium',
				 color=dataset['color'][line],
				 weight='normal'
				).set_size(15)

	
	plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
	plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
	
	plt.title('Visualization for {}'.format(word.title()))
	plt.savefig(word + '.pdf')
	plt.show()
	
def print_most_similar(word):
	print(word)
	print_formatted_output(model.wv.most_similar(positive=[word],  topn=5))
	print("\n")

def print_formatted_output(output):
	print(''.join(x[0] + "  " + '{0:.4f}'.format(x[1]) + "\n" for x in output))
	
	
if __name__ == '__main__':
	model = Word2Vec.load("word2vec.model")
	
	print("Size of the vocab: ", len(model.wv.vocab))
	print(model.wv['hogwarts'])
	print(model.wv.index2entity[:100])
	
	print_most_similar("dumbledore")
	print_most_similar("harry")
	print_most_similar("voldemort")
	print_most_similar("muggle")
	print_most_similar("horcrux")
	print_most_similar("hogwarts")
		

	print(model.wv.similarity("harry", "ron"))
	print(model.wv.similarity("ron", "hermione"))
	print(model.wv.similarity("harry", "voldemort"))
	print(model.wv.similarity("expelliarmus", "kedavra"))
	print(model.wv.similarity("fred", "george"))
	print(model.wv.similarity("wand", "elder"))

	print(model.wv.doesnt_match(['lucius', 'harry', 'bellatrix', 'greyback']))
	print(model.wv.doesnt_match(['ron', 'harry', 'hermione', 'snape']))
	print(model.wv.doesnt_match(['expelliarmus', 'kedavra', 'lumos', 'magic']))
	print(model.wv.doesnt_match(['crookshanks', 'hedwig', 'fang', 'snake']))
	print(model.wv.doesnt_match(['pound', 'sickle', 'knut', 'galleon']))
	print(model.wv.doesnt_match(['potter', 'weasley', 'malfoy', 'neville']))
	
	print_formatted_output(model.wv.most_similar(positive=["draco", "ron"], negative=["harry"], topn=5))
	print_formatted_output(model.wv.most_similar(positive=["arthur", "vernon"], negative=["petunia"], topn=5))

	plot(model, "weasley")
	plot(model, "ginny",  [i[0] for i in model.wv.most_similar(negative=["ginny"], topn=10)])
	