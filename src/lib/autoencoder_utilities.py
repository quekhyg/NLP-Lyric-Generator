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

def mask_random_inputs(x_encoder, x_decoder, y, n_copies = 3, mask_prob = 0.15, mask_token = '<mask>', random_seed = 2022):
    rng = np.random.default_rng(seed = random_seed)
    masked_x_encoder = []
    masked_x_decoder = []
    masked_y = []
    for x_enc, x_dec, y_datum in zip(x_encoder, x_decoder, y):
        for n in n_copies:
            masked_x = [word if rng.random() < mask_prob else mask_token for word in x_enc]
            masked_x_encoder.append(masked_x)
            masked_x_decoder.append(x_dec)
            masked_y.append(y_datum)
    return masked_x_encoder, masked_x_decoder, masked_y
    
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

### Need to update ###
def ind_to_input_fun(indices, depth, to_mask = False, **kwargs):
    if to_mask:
        indices = mask_last([indices], vocab_to_index_dict = vocab_to_index, mask_token = mask_token)[0]
    input_oh = tf.one_hot(indices, depth = depth)
    x_enc = tf.expand_dims(input_oh, 0)
    x_dec = tf.identity(x_enc)
    return [x,x]
### Need to update ###
def update_input_fun(curr_input, pred_index, depth, to_mask = False, **kwargs):
    #assert curr_input[0] == curr_input[1], 'Error: input to encoder and decoder are different'
    x = curr_input[0]
    if to_mask:
        x_enc = x[:,1:-1,:]
        pred_oh = tf.one_hot([pred_index, mask_index], depth = depth)
    else:
        x_enc = x[:,1:,:]
        pred_oh = tf.one_hot(pred_index, depth = depth)
    input_index = tf.expand_dims([pred_oh], 0)
    x_enc = tf.concat([x[:,1:,:],input_index], 1)
    x_dec = tf.identity(x_enc)
    if to_mask:
        x_enc[:,
    return [x,x]