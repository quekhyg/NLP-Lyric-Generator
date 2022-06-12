import re
import os
from collections import Counter
import numpy as np
import tensorflow.random as rnd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_lg")

def load_corpus(path = '../../data', end_song_token = '\n\n<EOS>\n\n'):
    """Loads corpus from filepath

    Args: 
      path (str): string specifying the filepath
      end_song_token (str): custom token indicating the end of each song
    
    Returns:
      corpus (str) string containing the corpus, with each song separated by the end_song_token
    """
    corpus = ''
    all_files = os.listdir(path)
    for file in all_files:
        with open(os.path.join(path, file)) as f:
            text = f.read()
            corpus += text
        corpus += end_song_token
    return corpus

def split_corpus(path = '../../data', val_split = 0.2, end_song_token = '\n\n<EOS>\n\n', random_seed = 2022):
    all_files = os.listdir(path)
    rng = np.random.default_rng(seed = random_seed)
    rng.shuffle(all_files)
    n = len(all_files)
    train_files = all_files[:int((1-val_split)*n)]
    val_files = all_files[int((1-val_split)*n):]
    train_corpus, val_corpus = '',''
    for file in train_files:
        with open(os.path.join(path, file)) as f:
            train_corpus += f.read()
        train_corpus += end_song_token
    for file in val_files:
        with open(os.path.join(path, file)) as f:
            val_corpus += f.read()
        val_corpus += end_song_token
    return train_corpus, val_corpus, train_files, val_files

def decontraction(text, **kwargs):
    """Expand most common contractions from string

    Args: 
      text (str): string containing contractions to expand
      **kwargs : Unused, included for compatibility with preprocess_text function
    
    Returns:
      text (str) string with contractions expanded
    """

    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    return text

def to_lower(text, **kwargs):
    return text.lower()

def remove_punct(text, keep=None):
    """Remove punctuations from text

    Args: 
      text (str): string containing punctuations to remove
      keep (str): string containing punctuations to keep
          e.g., '\!|\?' removes all punctuations except "!" and "?"
    
    Returns:
      text (str) string with contractions expanded
    """
    
    from string import punctuation

    if keep:
        punctuation = re.sub(keep, '', punctuation)
    table=str.maketrans('','',punctuation)
    text = text.translate(table)
    text = re.sub(' +', ' ', text)
    
    return text


def remove_emoji(text, **kwargs):
    """Remove punctuations from text

    Args: 
      text (str): string containing emojis to remove
      **kwargs : Unused, included for compatibility with preprocess_text function
    
    Returns:
      text (str) string with emojis expanded
    """
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)


def remove_url(text, **kwargs):
    """Remove URLs from text

    Args: 
      text (str): string containing URLs to remove
      **kwargs : Unused, included for compatibility with preprocess_text function
    
    Returns:
      text (str) string with URLs expanded
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())

def split_text(text, delim = None, level = 'song'):
    """Converts text into a list of texts

    Args:
      text (str): string to be split
      delim (str or None): delimiter to split the text by. if not provided, level will be used to determine delimiter
      level ('song', 'verse', 'line', 'word'): 4 user-defined options, which determines the default delimiter if delim is not provided.
    
    Returns:
      text (list) list of strs from the text split by delimiter
    """
    if delim is None:
        if level == 'song':
            delim = '\n\n<EOS>\n\n'
        elif level == 'verse':
            delim = '(<BRIDGE>|<CHORUS>|<OTHERS>|<PRECHORUS>|<PRELUDE>|<VERSE>)'
        elif level == 'line':
            delim = '\n+'
        elif level == 'word':
            delim = '\s+'
        else:
            print('Error: Since no delimiter was provided, level argument should be one of these options: \'song\', \'verse\', \'line\', \'word\'')
            return None
    return re.split(delim, text)

def split_song(song_text, verse_types = ['<BRIDGE>','<CHORUS>','<OTHERS>','<PRECHORUS>','<PRELUDE>','<VERSE>'], stripwhite = True):
    """Converts song text into a dictionary of each type of verse

    Args:
      song_text (str): song string to be split
      verse_types (list of tokens): list of standard annotated tokens indicating the type of verse
      stripwhite (bool): whether to remove whitespace at each end of each verse
    
    Returns:
      verses (dict) dictionary. Each key corresponds to a type of verse, each value is a list of strings, with each string corresponding to each instance of that verse type. E.g. <VERSE>: [verse1str, verse2str, verse3str]
    """           
    all_verses = {}
    for verse_type in verse_types:
        verses = re.findall(verse_type+'(.+?)<', song_text, re.DOTALL)
        if stripwhite:
            verses = [x.strip() for x in verses]
        all_verses[verse_type] = verses
    return all_verses

def remove_annotation(text, annotation = '<.+>'):
    return re.sub(annotation, '', text)

def preprocess_text(text, fun_list = [to_lower, decontraction, remove_punct, remove_emoji, remove_url], **kwargs):
    """Performs standard preprocessing functions on text

    Args: 
      text (str): string containing text to be processed
      fun_list (list of functions) : List of functions to be sequentially performed on the text
      **kwargs (args) : Keyword arguments to be passed into each function in fun_list
    
    Returns:
      text (str) string of processed text
    """
    for fun in fun_list:
        text = fun(text, **kwargs)
    return text

def tokenize_text(text, newline_token):
    words = preprocess_text(text, fun_list = [to_lower, decontraction, remove_punct], keep = '\<|\>')
    words = re.sub(r'\n',f' {newline_token} ', words)
    words = re.split(' +', words) #Tokenising
    return words

def tokenize_corpus(corpus_text, window_length,
                    index_to_vocab = None, vocab_to_index = None,
                    end_token = '<eos>', start_token = '<cls>',
                    pad_token = '<pad>', unk_token = '<unk>',
                    newline_token = '<new>'):
    words = tokenize_text(corpus_text, newline_token)
    
    word_count = Counter(words)
    for special_token in [end_token, start_token, pad_token, unk_token]:
        if special_token not in word_count:
            word_count[special_token] = 0

    #Reference Dictionaries to convert one-hot index to string and vice versa
    if index_to_vocab is None:
        index_to_vocab = {i: k for i, k in enumerate(word_count.keys())}
    if vocab_to_index is None:
        vocab_to_index = {k: i for i, k in enumerate(word_count.keys())}

    songs = ' '.join(words)
    songs = songs.split(f' {newline_token} {newline_token} {end_token} {newline_token} {newline_token} ')
    songs = [song.split(' ') for song in songs]
    songs = [[pad_token]*(window_length-1) + [start_token] + song + [end_token] + [pad_token]*(window_length-1) for song in songs]
    songs_token_ind = [[vocab_to_index.get(x) for x in song] for song in songs]
    
    return words, word_count, index_to_vocab, vocab_to_index, songs, songs_token_ind


def generate_text(model,
                  ind_to_input_fun, update_input_fun,
                  start_string,
                  window_length,
                  vocab_to_index_dict, index_to_vocab_dict,
                  vocab_size = None,
                  num_generate = 100, temperature = 1.0,
                  random_seed = 2022,
                  end_token = '<eos>', start_token = '<cls>',
                  pad_token = '<pad>', unk_token = '<unk>',
                  newline_token = '<new>',
                  **kwargs):
    if vocab_size is None:
        vocab_size = max(vocab_to_index_dict.values())
    # Converting our start string to numbers (vectorizing).
    tokenized_str = tokenize_text(start_string, newline_token)
    input_indices = [vocab_to_index_dict.get(s) for i, s in enumerate(tokenized_str) if i < window_length-1]
    input_indices = [i if i is not None else vocab_to_index_dict.get(unk_token) for i in input_indices]
    input_indices = [vocab_to_index_dict.get(pad_token)]*(window_length - len(input_indices)-1) + [vocab_to_index_dict.get(start_token)] + input_indices
    
    model_input = ind_to_input_fun(input_indices, **kwargs)

    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
    model.reset_states()
    for word_index in range(num_generate):
        prediction = model.predict(model_input)

        # Using a categorical distribution to predict the character returned by the model.
        prediction = prediction / temperature
        predicted_id = rnd.categorical(np.log(prediction), num_samples=1, seed = random_seed)[-1,0]
        
        # Updating model input with new predicted word
        model_input = update_input_fun(model_input, predicted_id, **kwargs)
        
        pred_word = index_to_vocab_dict[predicted_id.numpy()]
        text_generated.append(pred_word)
        if pred_word == end_token:
            break
    
    return (start_string + ' ' + ' '.join(text_generated)), text_generated


def find_cossim(generated_text, corpus_text):
    """Find cosine similarity between generated text and full / validation corpus.

    Args: 
      generated_text (str): Generated lyrics by lyrics generation model
      corpus_text (str) : Full Corpus from load_corpus function or Validation Corpus from split_corpus function
    
    Returns:
      cosine_similarity (float) : Cosine similarity between the vectors of the generated lyrics and the corpus
    """
    gen_text = nlp(generated_text)
    corp_text = nlp(corpus_text)
    return cosine_similarity([gen_text.vector], [corp_text.vector])[0][0]
