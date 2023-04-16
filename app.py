from flask import Flask, render_template, request
import openai

API_KEY = "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"


app = Flask(__name__)


openai.api_key = API_KEY 
openai.Model.list()

BASIC_PROMPT = "put me in this JSON{'symptoms:'} each symptom you see in the following sentence :"

welcome_msg = "Hey, welcome! I am the new generation of health ChatBot, let's ask questions about your health"
all_messages = [welcome_msg]

def generate_prompt(user_input: str) -> str :
    return f'{BASIC_PROMPT} {user_input}'


@app.route("/")
def default():
    return render_template("index.html", all_messages=all_messages)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    current_message = request.form['symptom-input']
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(current_message),
        temperature=0.1
    )
    all_messages.append(response)
    return render_template("index.html", all_messages=all_messages)

if __name__ == "__main__":
    app.run(debug=True)