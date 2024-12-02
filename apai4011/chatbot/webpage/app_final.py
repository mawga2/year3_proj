# Essential Libraries
import re
import string
import pickle
import torch
import spacy
import nltk
import pytextrank
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from zhipuai import ZhipuAI
from flask import Flask, render_template, request, jsonify
import os

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Load necessary files
clf_file = "../models/nb.pkl"
tv_file = "../models/tv_intent.pickle"
diag_file = '../models/finetuned_BERT_epoch_9.model'

def identitity(x):
    return x

# Load the intent classification model and TF-IDF transformer
with open(clf_file, 'rb') as file:
    model_intent = pickle.load(file)
tv = pickle.load(open(tv_file, "rb"))

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

# Text Preprocessing Functions
def convert_to_lowercase(text):
    return text.lower()

def emoji_to_word(tweet):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U0001F1F2-\U0001F1F4" 
        u"\U0001F1E6-\U0001F1FF" 
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
    for emot in Emoticon_Dictionary:
        tweet = re.sub(u'('+emot+')', "_".join(Emoticon_Dictionary[emot].replace(",","").split()), tweet)
    return tweet

def remove_pattern(tweet, pattern):
    tweet = re.sub(pattern,'',tweet)
    return tweet

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_user_mentions(text):
    return re.sub(r"@\w+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join(word for word in text.split() if word not in STOPWORDS)

def preprocess_pipeline(text):
    text = remove_urls(text)
    text = emoticons_to_word(text)
    text = emoji_to_word(text)
    text = remove_user_mentions(text)
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatization(text):
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())

def tokenization(text):
    return word_tokenize(text)

# Prediction Helpers
def transform(tokens):
  return(tv.transform([tokens]).toarray())

def predict_from(text, model):
  tokens = tokenization(lemmatization(preprocess_pipeline(text)))
  if (len(tokens) == 1) and text.lower()[0:4] == "what":
    return 1
  return model.predict(transform(tokens))

# Device Setup for Torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Diagnosis Model
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
 'Urinary tract infection': 18,
 'Allergy': 19,
 'Gastroesophageal reflux disease': 20,
 'Drug reaction': 21,
 'Peptic ulcer disease': 22,
 'Diabetes': 23}

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
 'Urinary tract infection',
 'Allergy',
 'Gastroesophageal reflux disease',
 'Drug reaction',
 'Peptic ulcer disease',
 'Diabetes']

model_diagnosis = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict), output_attentions=False, output_hidden_states=False)
model_diagnosis.to(device)
model_diagnosis.load_state_dict(torch.load(diag_file, map_location=device))

# Tokenizer & Pipeline for Diagnosis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
pipe = TextClassificationPipeline(model=model_diagnosis, tokenizer=tokenizer, return_all_scores=True)

# SpaCy Setup for TextRank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract(text):
    doc = nlp(text)
    keywords = [phrase.text for phrase in doc._.phrases]
    keywords = [remove_stopwords(keyword) for keyword in keywords]
    return keywords[0] if keywords else None

# Intent Detection Function
def NLU(text):
    intent = predict_from(text, model_intent)
    if intent == 0:
        result = pipe(text)
        df_result = pd.DataFrame(result[0])
        disease_index = df_result.score.idxmax()
        return 0, label_list[disease_index]
    if intent == 1:
        target = extract(text)
        return 1, target
    else:
        return 2, None

app = Flask(__name__)

# Set up the file paths for the CSVs
CSV_PATH_1 = "../knowledge_base/kaggle_diseases_precautionNdescriptionNmedicine_v2.csv"
CSV_PATH_2 = "../knowledge_base/modified_2.csv"

# Function to read 1.csv (diagnosis-related content)    
def load_csv_1(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()
    
# Function to read 2.csv (information-related content)
def load_csv_2(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()

disease_definitions_df = load_csv_1(CSV_PATH_1)  # Content for diagnosis
info_definitions_df = load_csv_2(CSV_PATH_2)  # Content for information

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def chat():
    return render_template('chatbot_interface.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['message'].lower()

    # Using the NLU model to detect intent and entities
    try:
        intent, entities = NLU(user_message)
        if intent == 0: # Diagnosis intent
            matched = disease_definitions_df[disease_definitions_df['disease'].str.contains(entities, case=False, na=False)]
            if not matched.empty:
                response = (
                    f"{matched['disease'].values[0]}: {matched['short_description'].values[0]}\n\n"
                    f"Key facts: {matched['keyfacts'].values[0]}\n\n"
                    f"Precautions: {matched['precaution_1'].values[0]}, {matched['precaution_2'].values[0]}, "
                    f"{matched['precaution_3'].values[0]}, {matched['precaution_4'].values[0]}\n\n"
                    f"Medicines: {matched['medicine'].values[0]}"
                )
            else:
                response = f"I don't have information on {entities}."
        elif intent == 1: # Info intent
            entities = entities.lower()
            matched = info_definitions_df[info_definitions_df['key_words'].str.strip() == entities]

            if matched.empty: # Search within parentheses
                entities_in_parentheses = info_definitions_df['key_words'].apply(lambda x: re.search(r'\((.*?)\)', x))
                matched = info_definitions_df[entities_in_parentheses.apply(lambda x: x.group(1) if x else '').str.match(entities, case=False, na=False)]
            
            if matched.empty: # Search as part of content
                matched = info_definitions_df[info_definitions_df['key_words'].str.contains(entities, case=False, na=False)]

            if matched.empty: # If all fails, ping chatGLM
                matched = get_from_chatGLM(entities)
            else:
                response = (
                    f"{matched['key_words'].values[0]}: {matched['definition'].values[0]}\n\n"
                    f"If you want to know more information, you can visit Healthline.com: {matched['url'].values[0]}"
                )
        else:
            response = f"I'm not sure how to help with '{user_message}'. Could you clarify?"

    except Exception as e:
        response = f"An error occurred: {str(e)}"
    
    return jsonify({"response": response})

client = ZhipuAI(api_key="6026d5961c40106882cd6848016ed219.06LGFZj3PNth6GmI")

def get_from_chatGLM(entities):
    try:
        # Define the task for ChatGLM
        messages = [
            {
                "role": "user",
                "content": f"Your task is to provide information on the following query: {entities}"
            }
        ]
        
        # Call the ChatGLM API
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=messages,
        )
        
        # Extract and return the response
        output_message = response.choices[0].message.content.strip()
        return output_message if output_message else "Sorry, I couldn't find relevant information."
    
    except Exception as e:
        return f"Error while querying ChatGLM: {str(e)}"

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    # Process the message (e.g., store in database, send email)
    print(f"Received message from {name} ({email}): {message}")

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(debug=True)