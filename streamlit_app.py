import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

# -------- Page Config --------
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)

# -------- Model Loading --------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("Anu2711/bert-disaster-tweets")
    model = DistilBertForSequenceClassification.from_pretrained("Anu2711/bert-disaster-tweets")
    model.eval()
    return tokenizer, model

# -------- Preprocessing --------
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'[URL]', text)

def remove_mentions(text):
    pattern = re.compile(r'@[A-Za-z_]\w*')
    return pattern.sub(r'[MENTION]', text)

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

def preprocess_text(text):
    text = remove_url(text)
    text = remove_mentions(text)
    text = clean_text(text)
    return text

# -------- Inference --------
def classify_tweet(tweet, tokenizer, model):
    tweet_cleaned = preprocess_text(tweet)
    encoding = tokenizer.encode_plus(
        tweet_cleaned,
        add_special_tokens=True,
        max_length=84,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        pred = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.softmax(outputs.logits, dim=1)
    confidence = probs[0, pred].item()
    return pred, confidence

# -------- UI --------
st.title("🚨 Disaster Tweet Classifier")
st.markdown(
    "Uses a fine-tuned **DistilBERT** model (F1: 0.82) to predict whether a tweet "
    "is about a real disaster or not."
)
st.divider()

tweet_input = st.text_area(
    "Enter a tweet:",
    placeholder='e.g. "Massive wildfire spreading through the hills near Los Angeles"',
    height=100
)

if st.button("Classify", type="primary"):
    if not tweet_input.strip():
        st.warning("Please enter a tweet before classifying.")
    else:
        with st.spinner("Classifying..."):
            tokenizer, model = load_model()
            pred, confidence = classify_tweet(tweet_input, tokenizer, model)

        if pred == 1:
            st.error("**Disaster** — this tweet appears to be about a real disaster.")
        else:
            st.success("**Not a Disaster** — this tweet does not appear to be about a real disaster.")

        st.markdown(f"**Confidence:** {confidence:.1%}")
        st.progress(confidence)

st.divider()
st.caption("Model trained on the [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) dataset.")