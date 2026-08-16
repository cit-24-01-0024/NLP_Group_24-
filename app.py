"""Compare saved Logistic Regression and LSTM sentiment models."""
from pathlib import Path
import pickle, re, nltk, numpy as np, streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ROOT=Path(__file__).parent; MODEL_DIR=ROOT/"models"; MAXLEN=100
NAMES=["logistic_regression_model.pkl","lstm_model.h5","tfidf_vectorizer.pkl","tokenizer.json","label_encoder.pkl"]
st.set_page_config(page_title="SentiScope",page_icon="💬",layout="wide")
st.title("SentiScope")
st.caption("Amazon Review Sentiment Analyzer | Logistic Regression vs LSTM")

def ensure_nltk():
    for path,pkg in {"tokenizers/punkt":"punkt","tokenizers/punkt_tab":"punkt_tab","corpora/stopwords":"stopwords","corpora/wordnet":"wordnet"}.items():
        try:nltk.data.find(path)
        except LookupError:nltk.download(pkg,quiet=True)

@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing=[x for x in NAMES if not (MODEL_DIR/x).is_file()]
    if missing:raise FileNotFoundError("Missing from models/: "+", ".join(missing))
    with (MODEL_DIR/NAMES[0]).open("rb") as f:lr=pickle.load(f)
    lstm=load_model(MODEL_DIR/NAMES[1],compile=False)
    with (MODEL_DIR/NAMES[2]).open("rb") as f:tfidf=pickle.load(f)
    with (MODEL_DIR/NAMES[3]).open(encoding="utf-8") as f:tok=tokenizer_from_json(f.read())
    with (MODEL_DIR/NAMES[4]).open("rb") as f:encoder=pickle.load(f)
    return lr,lstm,tfidf,tok,encoder

@st.cache_resource
def language_tools():
    ensure_nltk()
    return set(stopwords.words("english")),WordNetLemmatizer()

def preprocess(text):
    text=re.sub(r"http\S+|www\S+","",str(text).lower())
    text=re.sub(r"[^a-z\s]","",text)
    text=re.sub(r"\s+"," ",text).strip()
    stops,lemma=language_tools()
    return " ".join(lemma.lemmatize(t) for t in word_tokenize(text) if t not in stops)

def decode(encoder,value):
    try:return str(encoder.inverse_transform([int(value)])[0])
    except (TypeError,ValueError):return str(value)

def predict_lr(text,model,vectorizer,encoder):
    x=vectorizer.transform([text]); pred=model.predict(x)[0]
    probabilities=np.asarray(model.predict_proba(x)[0])
    index=int(np.where(np.asarray(model.classes_)==pred)[0][0])
    return decode(encoder,pred),float(probabilities[index])

def predict_lstm(text,model,tokenizer,encoder):
    x=pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=MAXLEN)
    probabilities=np.asarray(model.predict(x,verbose=0)[0]); index=int(np.argmax(probabilities))
    return decode(encoder,index),float(probabilities[index])

try:lr,lstm,tfidf,tokenizer,encoder=load_artifacts()
except Exception as error:
    st.error("The model files could not be loaded.");st.code(str(error))
    st.info("Place all five artifacts in models/ beside app.py, then restart.");st.stop()

review=st.text_area("Enter an Amazon product review",height=150,placeholder="This product works perfectly. I love it!")
if st.button("Analyze sentiment",type="primary",use_container_width=True):
    if not review.strip():st.warning("Please enter a review.")
    else:
        try:
            cleaned=preprocess(review)
            if not cleaned:st.warning("No usable words remained after preprocessing.")
            else:
                left,right=st.columns(2)
                lr_label,lr_conf=predict_lr(cleaned,lr,tfidf,encoder)
                lstm_label,lstm_conf=predict_lstm(cleaned,lstm,tokenizer,encoder)
                with left:
                    st.subheader("Logistic Regression")
                    st.metric("Sentiment",lr_label);st.progress(lr_conf);st.write(f"Confidence: {lr_conf:.1%}")
                with right:
                    st.subheader("LSTM")
                    st.metric("Sentiment",lstm_label);st.progress(lstm_conf);st.write(f"Confidence: {lstm_conf:.1%}")
                st.success(f"Both models agree: {lr_label}") if lr_label==lstm_label else st.info("The models disagree because they represent text differently.")
                with st.expander("Show preprocessed text"):st.code(cleaned)
        except Exception as error:st.error("The review could not be analyzed.");st.exception(error)

with st.expander("How the models work"):
    st.markdown("- **Logistic Regression:** saved TF-IDF features.\n- **LSTM:** saved tokenizer; sequences padded to 100 tokens.\n- The saved encoder displays **Negative**, **Neutral**, or **Positive**.")
with st.expander("Limitations and responsible use"):
    st.write("Sarcasm, mixed opinions, slang, short reviews, and unfamiliar language may be misclassified. Class imbalance can make minority-class confidence misleading. Use for educational analysis, not consequential decisions.")
st.caption("Confidence is model probability, not a guarantee.")
