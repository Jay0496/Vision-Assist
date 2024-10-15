# Flask Chatterbot

This project is a Flask-based chat application powered by the ChatterBot library. I initially created this project following a tutorial and then expanded on its design to enhance the user experience.

## Usage
Once the app is running locally, you can interact with the bot by typing messages in the chatbox and receiving responses. The bot's responses are generated based on the corpus data from the ChatterBot library.

## Features:
Chatbox UI: A sleek, text-message-like interface where users can type and send messages.
Auto-scrolling: New messages automatically scroll into view.
Responsive Design: The interface adjusts well to different screen sizes, especially mobile-like layouts.

## Tools Used

- **Flask**: A lightweight web framework for Python.
- **ChatterBot**: A machine learning-based conversational dialogue engine.
- **jQuery**: A JavaScript library for easier DOM manipulation and event handling.
- **HTML/CSS**: For structuring and styling the web application.
- **Anaconda**: A distribution of Python and R for scientific computing and data science, which I used to manage my virtual environment.

## Getting Started

### Prerequisites

- Python 3.7
- Anaconda installed on your machine

### Installation

1. Clone this repository to your local machine
2. Navigate to the project directory
3. Create and activate a virtual environment (optional)
  conda create --name chatbox python=3.7
  conda activate chatbox
4. Install the required dependencies:
  pip install flask chatterbot chatterbot_corpus pytz
5. Run the application:
  python app.py
6. Open your web browser and go to http://127.0.0.1:5000 to interact with the chatbot.

### Additional Commands
1. To deactivate the environment:
  conda deactivate
2. To remove the environment if needed:
  conda remove --name chatbox --all

## Next Steps and Expansion
The next steps for this project are to improve the chatbot's ability to give better responses based on the prompts it receives.
