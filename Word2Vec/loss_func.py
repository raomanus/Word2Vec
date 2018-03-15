import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    # batch_size of the context word embeddings
    batch_size = inputs.get_shape().as_list()
    batch_size = batch_size[0]
    # taking the dot product of the context embedding and the target embeddings
    temp_tensor = tf.reduce_sum(tf.multiply(inputs, true_w), axis=1)
    # remove the nan values from the dot product result
    temp_tensor = tf.where(tf.is_nan(temp_tensor), tf.zeros_like(temp_tensor),temp_tensor)
    temp_tensor = tf.reshape(temp_tensor, [batch_size, 1])

    # taking log the dot product
    A = tf.log(tf.exp(temp_tensor))
    
    # calculating the score for each possible combination of context and target words.
    B = tf.reshape(tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, true_w, transpose_b=True)), axis=1)), [batch_size, 1])

    # returning the loss value.
    return tf.subtract(B,A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    # Extracting the sample, batch and embedding sizes.
    sample_size = len(sample)
    input_size = inputs.get_shape().as_list()
    batch_size = input_size[0]
    embedding_size = input_size[1]
    
    # creating additional tensors for further calculation
    one_matrix = [[1.0 for j in range(sample_size)] for i in range(batch_size)]
    one_tensor = tf.reshape(tf.convert_to_tensor(one_matrix, dtype=tf.float32), [batch_size,sample_size])

    # arrays of small corrections to be added to tensors
    small_values = [[0.0000000001 for j in range(sample_size)] for i in range(batch_size)]
    sample_additive2 = [0.0000000001 for j in range(batch_size)]

    # tensor values for the above arrays
    small_val_tensor = tf.reshape(tf.convert_to_tensor(small_values, dtype=tf.float32), [batch_size,sample_size])
    sample_additive2 = tf.reshape(tf.convert_to_tensor(sample_additive2, dtype=tf.float32), [batch_size, 1])

    # extracting the target word embeddings from the given weights.
    target_embeddings = tf.reshape(tf.nn.embedding_lookup(weights, labels, name="TargetWordEmbedding"), [batch_size, embedding_size]) # [batch_size, embedding_size]
    
    # extracting unigram probabilities for the target words.
    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    unigram_probabilities = tf.nn.embedding_lookup(unigram_prob, labels, name="TargetUnigramProbs") # [batch_size, 1]
    
    # extracting the target bias for the target words.
    target_bias = tf.nn.embedding_lookup(biases, labels, name="TargetBias") # [batch_size, 1]

    # calculating the dot product of context and target word embeddings
    sample_probs = tf.reshape(tf.reduce_sum( tf.multiply( inputs, target_embeddings ), axis=1 ), [batch_size, 1]) # [batch_size, 1]

    # convert negative words to tensor
    samples = tf.convert_to_tensor(sample, dtype=tf.int32)

    # calculating the log(kPr(x))
    unigram_probabilities = tf.log(tf.add(tf.scalar_mul(sample_size, unigram_probabilities), sample_additive2)) # [batch_size,1]

    # subtracting the unigram probabiities from the score and calculating the Pr(D=1 wo|wc)
    sample_probs = tf.subtract(sample_probs, unigram_probabilities) # [batch_size, 1]
    sample_probs = tf.sigmoid(sample_probs)
    sample_probs = tf.log(sample_probs) # [batch_size, 1]

    ###### End of first part ########

    # extracting the negative word embeddings from the given weights.
    negative_target_embeddings = tf.nn.embedding_lookup(weights, samples, name="NegativeTargetEmbedding") # [sample_size, embedding_size]
    
    # extracting unigram probabilities for the negative words.
    negative_unigram_probabilities = tf.transpose(tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample, name="NegativeUnigramProbs"), [sample_size, 1])) # [1, sample_size]
    
    # extracting the negative word bias and changing tensor shape for future calculations
    negative_target_bias = tf.reshape(tf.nn.embedding_lookup(biases, sample, name="NegativeTargetBias"), [sample_size, 1]) # [sample_size, 1]
    negative_target_bias = tf.transpose(negative_target_bias) # [1, sample_size]
    negative_target_bias = tf.tile(negative_target_bias,[batch_size, 1]) # [batch_size, sample_size]

    # calculating the dot product for the context word and negative word embeddings
    negative_sample_probs = tf.matmul(inputs, negative_target_embeddings, transpose_b=True) # [batch_size, sample_size]

    # adding negative word bias
    negative_sample_probs = tf.add(negative_sample_probs, negative_target_bias) # [batch_size, sample_size]

    # calculating the log(kPr{wx}) and changing the shape for future calculations
    negative_unigram_probabilities = tf.log(tf.scalar_mul(sample_size, negative_unigram_probabilities)) # [1, sample_size] 
    negative_unigram_probabilities = tf.tile(negative_unigram_probabilities,[batch_size, 1]) # [batch_size, sample_size]

    # calculating the final steps of the equation
    negative_sample_probs = tf.subtract(negative_sample_probs, negative_unigram_probabilities)
    negative_sample_probs = tf.subtract(one_matrix, tf.sigmoid(negative_sample_probs))
    negative_sample_probs = tf.reshape(tf.reduce_sum(tf.log(tf.add(negative_sample_probs, small_val_tensor)), axis=1), [batch_size, 1])

    sample_probs = tf.scalar_mul(-1, tf.add(sample_probs, negative_sample_probs))    

    return sample_probs
