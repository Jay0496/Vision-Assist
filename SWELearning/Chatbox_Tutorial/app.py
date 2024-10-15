from flask import Flask, render_template, request  # Import necessary Flask modules for routing and handling requests
from chatterbot import ChatBot  # Import ChatBot class from chatterbot library
from chatterbot.trainers import ChatterBotCorpusTrainer  # Import the trainer to train the chatbot

app = Flask(__name__)  # Initialize the Flask app

# Create a ChatBot instance with a SQL storage adapter
english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")

# Train the bot using the English corpus
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")  # Render the home page (index.html)

# Route to get bot responses
@app.route("/get", methods=['GET'])
def get_bot_response():
    userText = request.args.get('msg')  # Get the message from the user input
    return str(english_bot.get_response(userText))  # Return the bot's response as a string

# Main entry point for running the app
if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
