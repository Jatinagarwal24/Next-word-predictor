import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("mlp_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
vocab_size = len(tokenizer.word_index) + 1

# Get embedding dimension
embedding_dim = 64

# Title
st.title("🧠 Next Word Predictor using MLP")
st.markdown("Enter a phrase and get predicted next **k** words using your trained model.")

# User Inputs
input_text = st.text_input("Enter input text:")
context_len = st.slider("Context Length", 1, 10, 5)
num_words = st.slider("Number of words to generate", 1, 20, 5)
seed = st.number_input("Random Seed (optional)", min_value=0, value=42, step=1)

# Prediction logic
def predict_next_k_words(text, k, context_len, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    output_text = text.strip().lower().split()

    for _ in range(k):
        context = output_text[-context_len:]
        padded = tokenizer.texts_to_sequences([' '.join(context)])
        if len(padded[0]) < context_len:
            padded[0] = [0] * (context_len - len(padded[0])) + padded[0]
        x_input = np.array([padded[0]])

        y_pred = model.predict(x_input, verbose=0)[0]
        next_word_index = np.argmax(y_pred)
        next_word = tokenizer.index_word.get(next_word_index, "<OOV>")
        output_text.append(next_word)

    return " ".join(output_text)

# Run
if st.button("Generate"):
    if input_text.strip() == "":
        st.warning("Please enter some input text.")
    else:
        result = predict_next_k_words(input_text, num_words, context_len, seed)
        st.success("### Predicted Text:")
        st.markdown(f"> {result}")
