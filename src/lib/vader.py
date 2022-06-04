'''
Import packages
'''
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
nltk.download('vader_lexicon')

'''
Example on how to open the text file containing lyrics
'''

#lyrics=[]
#with open('we will rise stronger together.txt') as f:
#    lyrics.append(f.read())
#lyrics = list(map(lambda x:x.strip(),lyrics))

'''
Function to compute overall sentiment score on lyrics (using VADER)
'''

def vader_sentiment_score(lyrics):

    #Store lyrics in dataframe
    df = pd.DataFrame(lyrics, columns = ['lyrics'])

    #Create lists to store the different scores for each word
    negative = []
    neutral = []
    positive = []
    compound = []

    #Initialize the model
    sid = SentimentIntensityAnalyzer()

    #Iterate for each row of lyrics (if more than one row) and append the scores
    #Overall sentiment score is defined by compound score
    for i in df.index:
        scores = sid.polarity_scores(df['lyrics'].iloc[i])
        negative.append(scores['neg'])
        neutral.append(scores['neu'])
        positive.append(scores['pos'])
        compound.append(scores['compound'])

    #output: compound score    
    return compound
