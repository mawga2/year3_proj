# -*- coding: utf-8 -*-
"""NLU.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dB38wAeDMmLaTW5PZfFYWyul66TXWE10

#Import, Load and Help Functions
"""

import kagglehub
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("word_tokenize")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,  confusion_matrix
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,RidgeClassifierCV
from tabulate import tabulate
from math import log
from math import exp
import random
import pickle
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from transformers import TextClassificationPipeline
!pip3 install pytextrank

import spacy
import pytextrank

clf_file = "/content/drive/MyDrive/Colab Notebooks/APAI4011/nb.pkl" #model of intent classification
tv_file = "/content/drive/MyDrive/Colab Notebooks/APAI4011/tv_intent.pickle" #tf-idf transformer
diag_file = '/content/drive/MyDrive/Colab Notebooks/APAI4011/sym2dis/finetuned_BERT_epoch_9.model' #model for diagnosis
intent_representation = {0: "diagnosis", 1: "info", 2: "oos"}

def identitity(x):
  return x

with open(clf_file, 'rb') as file:
    model_intent = pickle.load(file)
tv = pickle.load(open(tv_file, "rb"))

def convert_to_lowercase(tweet):
    '''
    aim: change all tweets to lower case
    '''
    tweet = tweet.lower()
    return tweet

def emoji_to_word(tweet):
    '''
    aim: remove all the emoji in the tweets
    '''
    """
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

    tweet = emoji_pattern.sub(r'', tweet)
    return tweet

def emoticons_to_word(tweet):
    '''
    aim: based on the emoticon dictionary, replace all the emoticons to corresponding word
    The emoticon dictionary is provided in the next block
    '''
    for emot in Emoticon_Dictionary:
        tweet = re.sub(u'('+emot+')', "_".join(Emoticon_Dictionary[emot].replace(",","").split()), tweet)
    return tweet

def remove_pattern(tweet, pattern):
    '''
    aim: remove all the "@users" appears in the tweets
    '''
    tweet = re.sub(pattern,'',tweet)
    return tweet

#added
def remove_user(tweet):
    return remove_pattern(tweet, r"""(?:@[\w_]+)""")

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(tweet):
    '''
    aim: remove all the punctuation from the tweet given
    Punctuations are characters other than alphaters and digits.
    '''
    tweet = tweet.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    return tweet

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(tweet):
    '''
    aim: remove all stopwords in the tweets
    '''
    tweet = " ".join([word for word in str(tweet).split() if word not in STOPWORDS])
    return tweet

def remove_urls(tweet):
    '''
    aim: remove all the urls contained inside the tweets
    '''
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    tweet = url_pattern.sub(r'', tweet)
    return tweet

Emoticon_Dictionary = {
    u":‑\)":"smiley",
    u":\)":"smiley",
    u":-\]":"smiley",
    u":\]":"smiley",
    u":-3":"smiley",
    u":3":"smiley",
    u":->":"smiley",
    u":>":"smiley",
    u"8-\)":"smiley",
    u":o\)":"smiley",
    u":-\}":"smiley",
    u":\}":"smiley",
    u":-\)":"smiley",
    u":c\)":"smiley",
    u":\^\)":"smiley",
    u"=\]":"smiley",
    u"=\)":"smiley",
    u":‑D":"Laughing",
    u":D":"Laughing",
    u"8‑D":"Laughing",
    u"8D":"Laughing",
    u"X‑D":"Laughing",
    u"XD":"Laughing",
    u"=D":"Laughing",
    u"=3":"Laughing",
    u"B\^D":"Laughing",
    u":-\)\)":"happy",
    u":‑\(":"sad",
    u":-\(":"sad",
    u":\(":"sad",
    u":‑c":"sad",
    u":c":"sad",
    u":‑<":"sad",
    u":<":"sad",
    u":‑\[":"sad",
    u":\[":"sad",
    u":-\|\|":"sad",
    u">:\[":"sad",
    u":\{":"sad",
    u":@":"sad",
    u">:\(":"sad",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"happiness",
    u":'\)":"happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"dismay",
    u"D;":"dismay",
    u"D=":"dismay",
    u"DX":"dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"smirk",
    u";\)":"smirk",
    u"\*-\)":"smirk",
    u"\*\)":"smirk",
    u";‑\]":"smirk",
    u";\]":"smirk",
    u";\^\)":"smirk",
    u":‑,":"smirk",
    u";D":"smirk",
    u":‑P":"playful",
    u":P":"playful",
    u"X‑P":"playful",
    u"XP":"playful",
    u":‑Þ":"playful",
    u":Þ":"playful",
    u":b":"playful",
    u"d:":"playful",
    u"=p":"playful",
    u">:P":"playful",
    u":‑/":"annoyed",
    u":/":"annoyed",
    u":-[.]":"annoyed",
    u">:[(\\\)]":"annoyed",
    u">:/":"annoyed",
    u":[(\\\)]":"annoyed",
    u"=/":"annoyed",
    u"=[(\\\)]":"annoyed",
    u":L":"annoyed",
    u"=L":"annoyed",
    u":S":"annoyed",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed",
    u":‑x":"tongue-tied",
    u":x":"tongue-tied",
    u":‑#":"tongue-tied",
    u":#":"tongue-tied",
    u":‑&":"tongue-tied",
    u":&":"tongue-tied",
    u"O:‑\)":"innocent",
    u"O:\)":"innocent",
    u"0:‑3":"innocent",
    u"0:3":"innocent",
    u"0:‑\)":"innocent",
    u"0:\)":"innocent",
    u":‑b":"cheeky",
    u"0;\^\)":"innocent",
    u">:‑\)":"Evil",
    u">:\)":"Evil",
    u"\}:‑\)":"Evil",
    u"\}:\)":"Evil",
    u"3:‑\)":"Evil",
    u"3:\)":"Evil",
    u">;\)":"Evil",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party",
    u"%‑\)":"confused",
    u"%\)":"confused",
    u":-###..":"sick",
    u":###..":"sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous",
    u"\(\^_\^;\)":"Nervous",
    u"\(-_-;\)":"Nervous",
    u"\(~_~;\) \(・\.・;\)":"Nervous",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"respect",
    u"_\(\._\.\)_":"respect",
    u"<\(_ _\)>":"respect",
    u"<m\(__\)m>":"respect",
    u"m\(__\)m":"respect",
    u"m\(_ _\)m":"respect",
    u"\('_'\)":"Sad",
    u"\(/_;\)":"Sad",
    u"\(T_T\) \(;_;\)":"Sad",
    u"\(;_;":"Sad",
    u"\(;_:\)":"Sad",
    u"\(;O;\)":"Sad",
    u"\(:_;\)":"Sad",
    u"\(ToT\)":"Sad",
    u";_;":"Sad",
    u";-;":"Sad",
    u";n;":"Sad",
    u";;":"Sad",
    u"Q\.Q":"Sad",
    u"T\.T":"Sad",
    u"QQ":"Sad",
    u"Q_Q":"Sad",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Laugh",
    u"<\^!\^>":"Laugh",
    u"\^/\^":"Laugh",
    u"\（\*\^_\^\*）" :"Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Laugh",
    u"\(^\^\)":"Laugh",
    u"\(\^\.\^\)":"Laugh",
    u"\(\^_\^\.\)":"Laugh",
    u"\(\^_\^\)":"Laugh",
    u"\(\^\^\)":"Laugh",
    u"\(\^J\^\)":"Laugh",
    u"\(\*\^\.\^\*\)":"Laugh",
    u"\(\^—\^\）":"Laugh",
    u"\(#\^\.\^#\)":"Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Cheerful",
    u"\(\^_\^\)v":"Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed"
}

def preprocess_pipeline(tweet):
    tweet = remove_urls(tweet)
    tweet = emoticons_to_word(tweet)
    tweet = emoji_to_word(tweet)
    tweet = remove_user(tweet)
    tweet = convert_to_lowercase(tweet)
    tweet = remove_punctuation(tweet)
    tweet = remove_stopwords(tweet)
    return tweet

lemmatizer = WordNetLemmatizer()
def lemmatization(tweet):
    '''
    aim: perform lemmatization on the text
    '''
    return " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])

def tokenization(tweet):
    '''
    aim: perform tokenization on the text
    '''
    return word_tokenize(tweet)

def transform(tokens):
  return(tv.transform([tokens]).toarray())
def predict_from(text, model):
  tokens = tokenization(lemmatization(preprocess_pipeline(text)))
  if (len(tokens) == 1) and text.lower()[0:4] == "what":
    return 1
  return model.predict(transform(tokens))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#disease to index
label_dict = {'Psoriasis': 0,
 'Varicose Veins': 1,
 'Typhoid': 2,
 'Chicken pox': 3,
 'Impetigo': 4,
 'Dengue': 5,
 'Fungal infection': 6,
 'Common Cold': 7,
 'Pneumonia': 8,
 'Dimorphic Hemorrhoids': 9,
 'Arthritis': 10,
 'Acne': 11,
 'Bronchial Asthma': 12,
 'Hypertension': 13,
 'Migraine': 14,
 'Cervical spondylosis': 15,
 'Jaundice': 16,
 'Malaria': 17,
 'urinary tract infection': 18,
 'allergy': 19,
 'gastroesophageal reflux disease': 20,
 'drug reaction': 21,
 'peptic ulcer disease': 22,
 'diabetes': 23}

#index to disease
label_list = ['Psoriasis',
 'Varicose Veins',
 'Typhoid',
 'Chicken pox',
 'Impetigo',
 'Dengue',
 'Fungal infection',
 'Common Cold',
 'Pneumonia',
 'Dimorphic Hemorrhoids',
 'Arthritis',
 'Acne',
 'Bronchial Asthma',
 'Hypertension',
 'Migraine',
 'Cervical spondylosis',
 'Jaundice',
 'Malaria',
 'urinary tract infection',
 'allergy',
 'gastroesophageal reflux disease',
 'drug reaction',
 'peptic ulcer disease',
 'diabetes']

model_diagnosis = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model_diagnosis.to(device)

model_diagnosis.load_state_dict(torch.load(diag_file, map_location=torch.device('cpu')))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

pipe = TextClassificationPipeline(model=model_diagnosis, tokenizer=tokenizer, return_all_scores=True)

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")

def extract(x):
  doc = nlp(x)
  if len(doc._.phrases) > 0:
    return doc._.phrases[0].text
  else:
    return None

"""#Natural Language Understanding and Samples"""

def NLU(text):
  intent = predict_from(text, model_intent)
  if intent == 0:
    result = pipe(text)
    df_result = pd.DataFrame(result[0])
    disease_index = df_result.score.idxmax()
    return {
        "intent": intent_representation[0],
        "disease": label_list[disease_index]
    }
  if intent == 1:
    target = extract(text)
    return {
        "intent": intent_representation[1],
        "target": target
    }
  return{
      "intent": intent_representation[2]
  }

NLU("I am having low-grade fever and running noses")

NLU("What do you mean by fever?")

NLU("I sell medicines for fever. I will give you some discount")