from collections import Counter
import tensorflow as tf
import numpy as np

def construct_seq_data(songs_token_ind, window_length):
    x_encoder = []
    x_decoder = []
    y = []

    for song in songs_token_ind:
        for i in range(len(song)-window_length):
            x_encoder.append(song[(i+1):(i+window_length+1)])
            x_decoder.append(song[i:(i+window_length)])
            y.append(song[i+window_length])
    return x_encoder, x_decoder, y

def mask_last(x_encoder, vocab_to_index_dict, mask_token = '<mask>'):
    return [x[:-1] + [vocab_to_index_dict[mask_token]] for x in x_encoder]

def mask_random_inputs(x, mask_prob = 0.1, mask_index = None, random_seed = 2022):
    rng = np.random.default_rng(seed = random_seed)
    return [word if rng.random() > mask_prob else mask_index for word in x]

def construct_song_seq(songs_token_ind, max_len, pad_index, n_copies = 10, mask_prob = 0.1,
                       mask_index = None,
                       start_index = None,
                       end_index = None,
                       random_seed = 2022):
    x_encoder = []
    x_decoder = []
    y = []
    for n in range(n_copies):
        for i in range(len(songs_token_ind)):
            x = mask_random_inputs(songs_token_ind[i],
                                    mask_prob = mask_prob,
                                    mask_index = mask_index,
                                    random_seed = random_seed)
            x = [start_index] + x + [end_index]
            x_padded = x + [pad_index]*(max_len-len(x)-2)
            for i, word in enumerate(x[:-1]):
                x_encoder.append(x_padded)
                x_decoder.append(word)
                y.append(x[i+1])

    return x_encoder, x_decoder, y
    
def construct_datasets(x_encoder, x_decoder, y, batch_size, buffer = 10000, random_seed = 2022, one_hot = True, vocab_size = None):
    dataset = tf.data.Dataset.from_tensor_slices(((x_encoder, x_decoder), y))
    dataset = dataset.shuffle(buffer_size = buffer, seed = random_seed)
    if one_hot:
        if vocab_size is None:
            print('Error: Please provide vocab size for one hot encoding')
            return None
        dataset = dataset.map(lambda x, y: ((tf.one_hot(x[0], depth = vocab_size), tf.one_hot(x[1], depth = vocab_size)),
                                         tf.one_hot(y, depth = vocab_size)))

    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset   

def ind_to_input_fun(indices, depth, to_mask = False, mask_index = None, **kwargs):
    input_oh_dec = tf.one_hot(indices, depth = depth)
    x_dec = tf.expand_dims(input_oh_dec, 0)
    
    if to_mask:
        indices_enc = indices[1:]+[mask_index]
        input_oh_enc = tf.one_hot(indices_enc, depth = depth)
        x_enc = tf.expand_dims(input_oh_enc, 0)
    else:
        x_enc = tf.identity(x_dec)

    return [x_enc, x_dec]

def update_input_fun(curr_input, pred_index, depth, to_mask = False, mask_index = None):
    pred_oh = tf.one_hot(pred_index, depth = depth)
    
    x_dec = curr_input[1]
    x_dec = x_dec[:,1:,:]
    input_index_dec = tf.expand_dims([pred_oh], 0)
    x_dec = tf.concat([x_dec,input_index_dec], 1)
    
    if to_mask:
        x_enc = curr_input[0]
        x_enc = x_enc[:,1:-1,:]
        pred_oh_enc = tf.one_hot([pred_index, mask_index], depth = depth)
        input_index_enc = tf.expand_dims(pred_oh_enc, 0)
        x_enc = tf.concat([x_enc,input_index_enc], 1)
    else:
        x_enc = tf.identity(x_dec)
    return [x_enc, x_dec]