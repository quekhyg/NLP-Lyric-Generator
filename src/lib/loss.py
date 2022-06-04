import tensorflow as tf
from tensorflow.keras.losses import Loss

class VaderSentimentLoss(Loss):
    # sentiment_target is the VADER value that we want to train towards
    # y labels are assumed to be one-hot encoded
    # weights is a tensor of Vader values, with each value corresponding to each value in the vocabulary, as per the 1 hot encoding in y
    # alpha is a weighting between the standard cross entropy loss and the sentiment mse loss, it should be between 0 and 1. 1 means no sentiment, 0 means only sentiment.
    def __init__(self, sentiment_target, weights, alpha = 0.5):
        super(VaderSentimentLoss, self).__init__()
        self.sentiment_target = sentiment_target
        self.weights = weights
        self.alpha = alpha
        
    # Compute loss
    def call(self, y_true, y_pred):
        cce_loss = tf.keras.losses.CategoricalCrossentropy(y_true, y_pred)
        sentiment_tensor = tf.constant(self.sentiment_target, shape = tf.shape(y_pred))
        sentiment_mse_loss = tf.keras.losses.MeanSquaredError(self.weights * y_pred, sentiment_tensor)
        return cce_loss*self.alpha + (1-self.alpha)*sentiment_mse_loss

class CustomSentimentLoss(Loss):
    # sentiment_target is the sentiment value that we want to train towards
    # y labels are assumed to be one-hot encoded
    # weights is a tensor of Vader values, with each value corresponding to each value in the vocabulary, as per the 1 hot encoding in y
    # sentiment_function is a custom function which takes in a string and outputs a sentiment score
    # index_to_vocab is a dictionary that converts has index as keys and string as values
    # alpha is a weighting between the standard cross entropy loss and the sentiment mse loss, it should be between 0 and 1. 1 means no sentiment, 0 means only sentiment.
    def __init__(self, sentiment_target, sentiment_function, index_to_vocab, alpha = 0.5):
        super(CustomSentimentLoss, self).__init__()
        self.sentiment_target = sentiment_target
        self.sentiment_function = sentiment_function
        self.index_to_vocab = index_to_vocab
        self.alpha = alpha
        
    # Compute loss
    def call(self, y_true, y_pred):
        cce_loss = tf.keras.losses.CategoricalCrossentropy(y_true, y_pred)
        indices = tf.where(y_pred)
        sentiment_pred = tf.vectorized_map(lambda x: self.sentiment_function(self.index_to_vocab[x]), indices)
        sentiment_target = tf.constant(self.sentiment_target, shape = tf.shape(y_pred))
        sentiment_mse_loss = tf.keras.losses.MeanSquaredError(sentiment_target, sentiment_pred)
        return cce_loss*self.alpha + (1-self.alpha)*sentiment_mse_loss