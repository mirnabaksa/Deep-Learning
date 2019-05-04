import gensim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# DISCLAIMER: this code was taken and adapted to my dataset from https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d

def plot_similar_words(title, labels, embedding_clusters, word_clusters, a):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
	
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
						 
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
	model_gn = Word2Vec.load("word2vec.model")
	keys = ['harry', 'ron', 'hermione', 'draco', 'dumbledore', 'snape', 'hagrid', 'neville', 'ginny', 'sirius', 'voldemort']

	embedding_clusters = []
	word_clusters = []
	for word in keys:
		embeddings= []
		words = []
		for similar_word, _ in model_gn.most_similar(word, topn=40):
			words.append(similar_word)
			embeddings.append(model_gn[similar_word])
		embedding_clusters.append(embeddings)
		word_clusters.append(words)


	embedding_clusters = np.array(embedding_clusters)
	n, m, k = embedding_clusters.shape
	tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
	embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
	plot_similar_words('Similar words from Harry Potter books', keys, embeddings_en_2d, word_clusters, 0.7,'similar.png')