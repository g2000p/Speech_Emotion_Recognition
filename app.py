import pandas as pd 
from nltk.corpus import stopwords 
from flask import Flask, redirect, url_for, request, render_template,session
from tensorflow import keras
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import pickle
from flask_session import Session


from tensorflow.python.keras.backend import update

app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

def text_preprocess(text, stop_words=False):
  '''
  Accepts text (a single string) and
  a parameters of preprocessing
  Returns preprocessed text
  
  '''
  #new comment

  #nltk.download('punkt')
  #nltk.download('stopwords')
  STOPWORDS = set(stopwords.words('english'))
  # clean text from non-words
  text = re.sub(r'\W+', ' ', text).lower()

  # tokenize the text
  tokens = word_tokenize(text.lower())

  if stop_words:
    # delete stop_words
    tokens = [token for token in tokens if token not in STOPWORDS]

  return tokens

def predict(texts,model):
  
    emotions_to_labels = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
    labels_to_emotions = {j:i for i,j in emotions_to_labels.items()}  
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)

    texts_prepr = [text_preprocess(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(texts_prepr)
    pad = pad_sequences(sequences, maxlen=35)
    predictions = model.predict(pad)
    labels = np.argmax(predictions, axis=1)
    for i, lbl in enumerate(labels):
        if(i==0):
          print(f'\'{texts[i]}\' --> {labels_to_emotions[lbl]}')
          return labels_to_emotions[lbl]
  
@app.route('/', methods=['GET'])
def index():
    # Main page
    session['sentence'] = ""
    return render_template('index.html')
    
@app.route('/index', methods=['GET'])
def index2():
    # Main page
    return render_template('index.html')

@app.route('/finalresult', methods=['POST', "GET"])
def finalres():
    # Main page
    result=""
    fungive=['gg']
    fungive[0]=session['sentence']
    print(fungive)
    model = keras.models.load_model('model.h5')
    record=predict(fungive,model)
    session.pop('sentence',None)  
    return render_template("finalresult.html",record=record,update=fungive[0])

@app.route('/record', methods = ['POST', "GET"])
def record():
   record=True
   transcript=""
   if request.method == "POST":
      information = request.data
      text=information.decode("utf-8") 
      session['sentence']=text
      print(session['sentence'])

      record=False
      return render_template("result.html",record=record)
   else:     
      return render_template("result.html",record=record)



if __name__ == '__main__':  
    app.run(debug=True)


