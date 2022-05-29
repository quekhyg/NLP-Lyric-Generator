import re

def decontraction(text):
    """Expand most common contractions from string

    Args: 
      text (str): string containing contractions to expand
    
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


def remove_emoji(text):
    """Remove punctuations from text

    Args: 
      text (str): string containing emojis to remove
    
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


def remove_url(text):
    """Remove URLs from text

    Args: 
      text (str): string containing URLs to remove
    
    Returns:
      text (str) string with URLs expanded
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())


