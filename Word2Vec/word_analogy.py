import os
import pickle
import numpy as np
from scipy import spatial


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
input_file = open('word_analogy_dev.txt', 'r')
output_file = open('word_analogy_nce.txt', 'w')
result = ""

for line in input_file:
	line.strip()
	word_pairs = line.split("||")[1]
	word_tuples = word_pairs.strip().split(",")
	cosine_scores = []
	for tup in word_tuples:
		word1, word2 = tup.strip().split(":")
		word1 = word1[1:]
		word2 = word2[:-1]
		word_embedding1 = embeddings[dictionary[word1]]
		word_embedding2 = embeddings[dictionary[word2]]
		cosine_scores.append((1-spatial.distance.cosine(word_embedding1, word_embedding2)))

	max_idx = cosine_scores.index(max(cosine_scores))
	min_idx = cosine_scores.index(min(cosine_scores))
	
	result += word_pairs.strip().replace(","," ")+" "+word_tuples[max_idx].strip()+" "+word_tuples[min_idx]+"\n"

output_file.write(result)
output_file.close()

