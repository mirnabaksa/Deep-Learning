import pandas
import numpy as np
import time


if __name__ == '__main__':
	# Read whole dataset
	reviews = pandas.read_csv("kindle_reviews.csv", index_col=0)
	
	# Info about the distribution of overall ratings
	print(reviews['overall'].value_counts())
	
	# Drop unnecessary columns
	reviews = reviews.drop(['asin', 'reviewTime', 'reviewerID', 'reviewerName', 'unixReviewTime'], axis=1)
	
	# Convert 1-5 grade to negative/positive/neutral
	# 0 - negative, 1 - positive, 2 - neutral
	reviews['sentiment'] = reviews['overall'].apply(lambda x: 0 if x == 1 or x == 2 else 1 if x == 5 or x == 4 else 2)
	
	# Sort by helpfulness, take 5 000 samples from each class
	df = reviews.sort_values('helpful',ascending = False).groupby('sentiment').head(5000)
	
	# Info about the distribution of overall ratings after filtering
	print(df['sentiment'].value_counts())
	df.to_csv("dataset.csv")
	
	
	
	
	
	
