from flask import Flask, render_template, request
import openai
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


API_KEY = "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"
MODEL = keras.models.load_model("model")
SEQUENCES_PADDING = 11

df = pd.read_csv("./datasets/disease_medication_dataset.csv")

app = Flask(__name__)

openai.api_key = API_KEY

BASIC_PROMPT = "return me each symptom you see in the following sentence and separate them with a blank space and in lower case :"

welcome_msg = (
    "Hey welcome in your new health ChatBot, let's ask questions about your health"
)
all_messages = [welcome_msg]


def generate_prompt(user_input: str) -> str:
    return f"{BASIC_PROMPT} {user_input}"


def prediction(input: str) -> str:
    labels = df["disease"].iloc[:4].drop_duplicates().values.flatten().tolist()
    tokenizer = Tokenizer()
    seq_input = tokenizer.texts_to_sequences(input)
    padded_input = pad_sequences(seq_input, maxlen=SEQUENCES_PADDING)
    prediction = MODEL.predict(padded_input)
    print(labels)
    print(input, prediction, labels[np.argmax(prediction)])
    medications = find_medications(labels[np.argmax(prediction)])
    print(medications)
    return f"You have this Disease: {labels[np.argmax(prediction)]}, you should try these medications: {medications}"


def find_medications(disease: str) -> str:
    for i, row in df.iterrows():
        if row["disease"] == disease:
            print(row["medication"])
            return row["medication"]


@app.route("/")
def default():
    return render_template("index.html", all_messages=all_messages)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.form["symptom-input"] != "":
        current_message = request.form["symptom-input"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(current_message),
            temperature=0.1,
        )
        all_messages.append(f"symptoms:{response['choices'][0]['text']}")
        all_messages.append(prediction(response["choices"][0]["text"]))
    return render_template("index.html", all_messages=all_messages)


if __name__ == "__main__":
    app.run(debug=True)
