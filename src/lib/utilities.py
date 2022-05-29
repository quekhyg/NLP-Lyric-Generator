import re
import os

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
