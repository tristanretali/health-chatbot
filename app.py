from flask import Flask, render_template, request
import openai
import pandas as pd
import numpy as np
import utils
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.fileshare import ShareFileClient


app = Flask(__name__)

API_KEY = "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"
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=healthmodel;AccountKey=sePKfWGoct+Tm3NEZOiWWXsXpbhHc3wN5hhG0JkZygbR14698Vb/oM36nbziO1ZS2VJFwRJaivmh+AStQWrASg==;EndpointSuffix=core.windows.net"
SHARE_NAME = "ai-model"
FILE_PATH = "variables.data-00000-of-00001"
DEST_FILE = "./model/variables/variables.data-00000-of-00001"
df = pd.read_csv("./datasets/disease_medication_dataset.csv")
df_fit = utils.df_symptoms
Y = pd.get_dummies(df_fit["disease"]).values
tokenizer = utils.create_tokenizer()

openai.api_key = API_KEY

BASIC_PROMPT = "return me each symptom you see in the following sentence and separate them with a blank space and in lower case. If you don't see a symptom return me 'failed' in lower case:"
NO_DISEASE_MESSAGES = [
    "I can't estimate a disease, provide me better informations",
    "This Chatbot estimates sicknesses based on symptoms and nothing else. Please respect that and enter your symptoms",
    "Provide me more details on your symptoms",
]
HELP_COMMAND = "Hi, this chatbot is designed to give you your sickness based on your symptoms and give you the different medications who treat this disease. Becareful, it's always better to consult a doctor."

welcome_msg = "Welcome in your new Health ChatBot, enter your symptoms"
all_messages = []


def reinitialize_model_file(path: str):
    """
    Delete the content of the file

    Args:
        path (str): path of the file
    """
    try:
        open(path, "w").close()
    except IOError:
        print("Failure")


def load_model_file(
    connection_string: str, share_name: str, file_path: str, dest_file: str
):
    """


    Args:
        connection_string (str): connection string of the Azure file storage
        share_name (str): name of the Azure File storage
        file_path (str): path of the file in the Azure File storage
        dest_file (str): path of the file
    """
    try:
        file_client = ShareFileClient.from_connection_string(
            conn_str=connection_string,
            share_name=share_name,
            file_path=file_path,
        )

        with open(dest_file, "wb") as file_handle:
            # Download the file from Azure into a stream
            data = file_client.download_file()
            # Write the stream to the local file
            data.readinto(file_handle)

    except ResourceNotFoundError as ex:
        print("ResourceNotFoundError:", ex.message)


reinitialize_model_file(DEST_FILE)

load_model_file(CONNECTION_STRING, SHARE_NAME, FILE_PATH, DEST_FILE)

MODEL = keras.models.load_model("model")


def set_label(df_train, Y) -> list:
    """
    Put the disease at their right place in a list

    Args:
        df_train (DataFrame): the dataset who contain my diseases
        Y (ndarray): the Disease with the one hot encoder

    Returns:
        list: return the list of diseases at their right place for the predictions
    """
    # Create list with lenght = diseases number
    labels = list(
        range(len(df_train["disease"].drop_duplicates().values.flatten().tolist()))
    )
    # Fill the list with diseases at their right place
    for i in range(0, Y.shape[0], 250):
        labels[list(Y[i]).index(1)] = df_train["disease"][i]
    return labels


# Set the labels
LABELS = set_label(df_fit, utils.Y)


def generate_prompt(user_input: str) -> str:
    """
    Create the prompt who will be send to the OpenAI API

    Args:
        user_input (str): the symptoms of the user

    Returns:
        str: return the entire prompt
    """
    return f"{BASIC_PROMPT} {user_input}"


def prediction(input: str) -> str:
    """
    Give the input to the model and predict a disease

    Args:
        input (str): the symptoms

    Returns:
        str: return the disease predict by the model
    """
    seq_input = tokenizer.texts_to_sequences([input])
    padded_input = pad_sequences(seq_input, maxlen=utils.SEQUENCES_PADDING)
    prediction = MODEL.predict(padded_input)
    right_prediction = LABELS[np.argmax(prediction)]
    return right_prediction


def find_medications(disease: str) -> str:
    """
    Search the medication for a specific disease

    Args:
        disease (str): the disease predict by the model

    Returns:
        str: return the medications associate to the disease
    """
    for i, row in df.iterrows():
        if row["disease"] == disease:
            return row["medication"][:-2]
    return "There is no medications data for this disease"


@app.route("/")
def default():
    return render_template(
        "index.html", welcome_msg=welcome_msg, all_messages=all_messages
    )


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.form["symptom-input"] != "":
        current_message = request.form["symptom-input"]
        if (
            current_message == "help"
            or current_message == "Help"
            or current_message == "HELP"
        ):
            all_messages.append(current_message)
            all_messages.append(HELP_COMMAND)
        else:
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
                    NO_DISEASE_MESSAGES[random.randint(0, len(NO_DISEASE_MESSAGES) - 1)]
                )
            else:
                all_messages.append(
                    f"You potentially have a {prediction(current_response)}. You should try these medications: {find_medications(prediction(current_response))}"
                )
    return render_template(
        "index.html", welcome_msg=welcome_msg, all_messages=all_messages
    )


if __name__ == "__main__":
    app.run(debug=True)
