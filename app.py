# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# @st.cache_resource
# def load_all():
#     model = load_model("lstm_model.h5")

#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)

#     with open("max_len.pkl", "rb") as f:
#         max_len = pickle.load(f)

#     return model, tokenizer, max_len

# model, tokenizer, max_len = load_all()

# # Reverse word index
# index_to_word = {index: word for word, index in tokenizer.word_index.items()}
# index_to_word[0] = ""
# # Prediction function
# def predict_next_words(text, num_words=5):
#     for _ in range(num_words):
#         text_input = text.lower()
#         seq = tokenizer.texts_to_sequences([text_input])
#         seq = pad_sequences(seq, maxlen=max_len, padding='pre')

#         pred = model.predict(seq, verbose=0)
#         top_indices = pred[0].argsort()[-3:]
#         pred_index = np.random.choice(top_indices)
        

#         next_word = index_to_word.get(pred_index, "")

#         if next_word == "":
#             break

#         text += " " + next_word

#     return text
# # UI
# st.set_page_config(page_title="Next Word Predictor", page_icon="🤖")

# st.title("Next Word Prediction (LSTM) 🤖")
# st.write("Type a sentence and get the next word prediction.")

# user_input = st.text_input("Enter text:")

# num_words = st.slider("Number of words to generate", 1, 10, 5)

# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text")
#     else:
#         generated_text = predict_next_words(user_input, num_words)
#         st.success("Generated Text:")
#         st.write(generated_text)
import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load files
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Reverse index
index_to_word = {index: word for word, index in tokenizer.word_index.items()}
index_to_word[0] = ""

def predict_next_words(text, num_words):
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text.lower()])
        seq = pad_sequences(seq, maxlen=max_len, padding='pre')

        pred = model.predict(seq, verbose=0)
        top_indices = pred[0].argsort()[-3:]
        pred_index = np.random.choice(top_indices)

        next_word = index_to_word.get(pred_index, "")
        if next_word == "":
            break

        text += " " + next_word

    return text

# UI
interface = gr.Interface(
    fn=predict_next_words,
    inputs=[
        gr.Textbox(label="Enter text"),
        gr.Slider(1, 10, value=5, label="Number of words")
    ],
    outputs="text",
    title="Next Word Predictor (LSTM)",
    description="Generate text using LSTM model"
)

interface.launch()