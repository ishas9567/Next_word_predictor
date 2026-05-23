import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Next Word Predictor", page_icon="✍️", layout="centered")

st.title("✍️ Next Word Predictor")
st.write("LSTM-powered sentence completion")

# Load assets
@st.cache_resource
def load_assets():
    from tensorflow.keras.models import load_model
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word

# Load with visible error if it fails
try:
    with st.spinner("Loading model..."):
        model, tokenizer, max_len, index_to_word = load_assets()
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# UI
seed_text = st.text_input("Enter seed text", placeholder="e.g. Life is")
num_words = st.slider("Number of words to predict", 1, 20, 5)

if st.button("Predict"):
    if not seed_text.strip():
        st.warning("Please enter some text.")
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        result = seed_text.strip().lower()
        try:
            for _ in range(num_words):
                seq = tokenizer.texts_to_sequences([result])[0]
                padded = pad_sequences([seq], maxlen=max_len, padding='pre')
                pred = model.predict(padded, verbose=0)
                pred_index = np.argmax(pred)
                next_word = index_to_word.get(pred_index, "")
                if not next_word:
                    break
                result += " " + next_word

            original_len = len(seed_text.strip().split())
            words = result.split()
            original = " ".join(words[:original_len])
            predicted = " ".join(words[original_len:])

            st.markdown("### Result")
            st.markdown(f"**{original}** :blue[{predicted}]")

        except Exception as e:
            st.error(f"Prediction error: {e}")
