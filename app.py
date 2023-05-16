from flask import Flask, render_template, request
import openai
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


API_KEY = "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"
MODEL = keras.models.load_model("model")
SEQUENCES_PADDING = 100
VOCAB_SIZE = 740
df = pd.read_csv("./datasets/disease_medication_dataset.csv").iloc[:250000]
df_fit = pd.read_csv("./datasets/final_dataset.csv").iloc[:250000]
Y = pd.get_dummies(df_fit["disease"]).values
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df_fit["symptoms"].values)

app = Flask(__name__)

openai.api_key = API_KEY

BASIC_PROMPT = "return me each symptom you see in the following sentence and separate them with a blank space and in lower case. If you don't see a symptom return me 'failed' in lower case:"

welcome_msg = (
    "Welcome in your new health ChatBot, let's ask questions about your health"
)
all_messages = []


def set_label(df_train, Y):
    # Create list with lenght = diseases number
    labels = list(
        range(len(df_train["disease"].drop_duplicates().values.flatten().tolist()))
    )
    # Fill the list with diseases at their right place
    for i in range(0, Y.shape[0], 250):
        labels[list(Y[i]).index(1)] = df_train["disease"][i]
    return labels


LABELS = set_label(df_fit, Y)


def generate_prompt(user_input: str) -> str:
    return f"{BASIC_PROMPT} {user_input}"


def prediction(input: str) -> str:
    seq_input = tokenizer.texts_to_sequences([input])
    padded_input = pad_sequences(seq_input, maxlen=SEQUENCES_PADDING)
    prediction = MODEL.predict(padded_input)
    right_prediction = LABELS[np.argmax(prediction)]
    return right_prediction


def find_medications(disease: str) -> str:
    for i, row in df.iterrows():
        if row["disease"] == disease:
            return row["medication"][:-2]


@app.route("/")
def default():
    return render_template(
        "index.html", welcome_msg=welcome_msg, all_messages=all_messages
    )


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.form["symptom-input"] != "":
        current_message = request.form["symptom-input"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(current_message),
            temperature=0.1,
        )
        if len(all_messages) == 6:
            del all_messages[:2]
        all_messages.append(current_message)
        current_response = response["choices"][0]["text"].replace("\n", "")
        if current_response == "failed":
            all_messages.append(
                "I can't estimate a disease, provide me better informations"
            )
        else:
            all_messages.append(
                f"You potentially have the Disease {prediction(current_response)}. \n You should try these medications: {find_medications(prediction(current_response))}"
            )
    return render_template(
        "index.html", welcome_msg=welcome_msg, all_messages=all_messages
    )


if __name__ == "__main__":
    app.run(debug=True)
