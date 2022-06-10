from . import utilities as utils
import numpy as np
from numpy.linalg import norm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class Sentiment:
    """
    A class for comparing sentiment scores of original text and generated text

    ...
    Attributes
    ----------
    original_tokens : list
        list of tokens for original text (defaults to None)
    generated_tokens : list
        list of tokens for generated text (defaults to None)
    ----------------------------------------------------------------------
    """

    def __init__(self, original_tokens=None, generated_tokens=None):

        self.original_tokens = original_tokens
        self.generated_tokens = generated_tokens

    def clean_text(self, **kwargs):
        """Clean text by standardized preprocessing, tokenize, and remove stopwords if specified

        Args:
        original_text (str, optional): original text to be cleaned
        generated_text (str, optional): generated text to be cleaned
        remove_stopwords (bool): whether to remove stopwords or not
        stop_words (list): list of stopwords to be removed. Required if remove_stopwords is True
        """
        
        if kwargs.get('original_text'):
            original_new_text = utils.preprocess_text(kwargs['original_text'])
            original_tokens = original_new_text.split(" ")
            if kwargs.get('remove_stopwords'):
                original_tokens = [t for t in original_tokens if t not in kwargs['stop_words']]
            self.original_tokens = original_tokens

        if kwargs.get('generated_text'):
            generated_new_text = utils.preprocess_text(kwargs['generated_text'])
            generated_tokens = generated_new_text.split(" ")
            if kwargs.get('remove_stopwords'):
                generated_tokens = [t for t in generated_tokens if t not in kwargs['stop_words']]
            self.generated_tokens = generated_tokens


    @staticmethod
    def get_theme_vector(sentiment_themes, embedding, topn=10):
        """Compute the average vector for each given theme based on a specified word embedding

        Args:
        sentiment_themes (list): list of strings of the sentiment themes to compute
        embedding (gensim Word2VecKeyedVectors): the word embedding for extracting theme vectors
        topn (int): determine the top n similar words to the theme for extraction (defaults to 10)
        """

        # initialize
        all_theme_vectors = {}

        sentiment_themes = [theme.lower() for theme in sentiment_themes]
        for theme in sentiment_themes:
            most_sim = [x[0] for x in embedding.most_similar(theme, topn=topn)]
            theme_vector = np.mean(embedding[most_sim], axis=0)
            all_theme_vectors[theme] = theme_vector

        Sentiment.sentiment_themes = sentiment_themes
        Sentiment.all_theme_vectors = all_theme_vectors
        Sentiment.embedding = embedding
        Sentiment.topn = topn


    def score_word_vector_sentiment(self):
        """Compute the cosine similarity score of each text with each theme"""
        
        # check if original & generated tokens exist
        if not self.original_tokens or not self.generated_tokens:
            error_text = """ORIGINAL or GENERATED tokens does NOT exist.
            Either pass it into the class using Sentimentality(original_text=tokens, generated_text=tokens)
            or use the clean_text method. """
            raise AttributeError(error_text)

        # check if theme average vectors exist
        try: self.all_theme_vectors
        except AttributeError as error:
            # print(error)

            error_text = """No theme vector exists. Run get_theme_vector() to compute them.
            For more info, refer to help(Sentimentality.get_theme_vector)"""
            raise AttributeError(error_text)

        # initialize
        sentiment_scores = {'original':{}, 'generated':{}}

        # get mean word vector of original and generated text
        original_vectors = [self.embedding[t] for t in self.original_tokens if t in self.embedding]
        original_vectors_avg = np.mean(original_vectors, axis=0)
        generated_vectors = [self.embedding[t] for t in self.generated_tokens if t in self.embedding]
        generated_vectors_avg = np.mean(generated_vectors, axis=0)

        for theme, vector in self.all_theme_vectors.items():
            original_cossim = np.dot(original_vectors_avg,vector)/(norm(original_vectors_avg)*norm(vector))
            sentiment_scores['original'][theme] = original_cossim
            generated_cossim = np.dot(generated_vectors_avg,vector)/(norm(generated_vectors_avg)*norm(vector))
            sentiment_scores['generated'][theme] = generated_cossim

        self.word_vector_sentiment_scores = sentiment_scores


    def score_vader_sentiment(self):
        """Compute the vader sentiment scores of original & generated text """
        
        # check if original & generated tokens exist
        if not self.original_tokens or not self.generated_tokens:
            error_text = """ORIGINAL or GENERATED tokens does NOT exist.
            Either pass it into the class using Sentimentality(original_text=tokens, generated_text=tokens)
            or use the clean_text method. """
            raise AttributeError(error_text)

        # initialize
        sentiment_scores = {'original':{}, 'generated':{}}

        # get mean vader sentiment score for original & generated text
        sid = SentimentIntensityAnalyzer()

        sentiment_scores['original'] = sid.polarity_scores(' '.join(self.original_tokens))
        sentiment_scores['generated'] = sid.polarity_scores(' '.join(self.generated_tokens))

        self.vader_sentiment_scores = sentiment_scores