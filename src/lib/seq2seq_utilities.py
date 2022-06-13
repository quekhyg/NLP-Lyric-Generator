from collections import Counter
import tensorflow as tf
import numpy as np

def construct_seq_data(songs_token_ind, window_length):
    x = []
    y = []

    for song in songs_token_ind:
        for i in range(len(song)-window_length):
            x.append(song[i:(i+window_length)])
            y.append(song[i+window_length])
    return x, y
    
def construct_datasets(x, y, batch_size, buffer = 10000, random_seed = 2022, one_hot = True, vocab_size = None):
    dataset = tf.data.Dataset.from_tensor_slices(((x, x), y))
    dataset = dataset.shuffle(buffer_size = buffer, seed = random_seed)
    if one_hot:
        if vocab_size is None:
            print('Error: Please provide vocab size for one hot encoding')
            return None
        dataset = dataset.map(lambda x, y: ((tf.one_hot(x[0], depth = vocab_size), tf.one_hot(x[1], depth = vocab_size)),
                                         tf.one_hot(y, depth = vocab_size)))

    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset   

def ind_to_input_fun(indices, depth, **kwargs):
    input_oh_dec = tf.one_hot(indices, depth = depth)
    x = tf.expand_dims(input_oh_dec, 0)
    return [x, x]

def update_input_fun(curr_input, pred_index, depth, **kwargs):
    pred_oh = tf.one_hot(pred_index, depth = depth)
    x = curr_input[1]
    x = x[:,1:,:]
    input_index_dec = tf.expand_dims([pred_oh], 0)
    x = tf.concat([x,input_index_dec], 1)

    return [x, x]