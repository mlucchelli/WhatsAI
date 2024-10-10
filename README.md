# WhatsAI
![image](https://github.com/user-attachments/assets/d18ded01-2597-467e-8cc1-338005d63723)


This application processes WhatsApp chat logs by cleaning up messages and generating structured conversations for training purposes. It converts unstructured chat data into a CSV format suitable for further use, such as training conversational models with T5. Additionally, it includes a Streamlit app for easy interaction.

> **⚠️ Alert**
> This is my first AI project, and I acknowledge that I have a lot to learn. I appreciate any feedback or suggestions to improve!

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Training Monitoring](#training-monitoring)
- [ChatBot UI](#chatbot-ui)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Cleans chat logs by removing unwanted content (e.g., omitted media).
- Processes messages into structured conversations for training models.
- Generates a CSV file with input-output pairs for conversational modeling.
- Handles large chat logs efficiently.
- Easy configuration with environment variables.
- Train a t5-base-spanish model with the logs
- Monitor the progress with tensorboard
- Setup a dummy chat app with Streamlit to play with your bot 

## Requirements
This application requires Python 3.8 or higher and the following packages:

- `transformers[torch]`: For training and using the T5 model with PyTorch support.
- `torch`: The deep learning framework used for model training.
- `streamlit`: For creating the interactive web interface.
- `python-dotenv`: To manage environment variables from a `.env` file.
- `sentencepiece`: For tokenization needed by T5 models.
- `numpy`: For numerical operations and data manipulation.
- `pandas`: For data handling and processing.
- `tensorboard`: For visualizing training metrics.

You can install all required libraries using:

    pip install -r requirements.txt

## Installation

1. Clone this repository:

    `git clone https://github.com/mlucchelli/whatsai-conversational-bot.git`

2. Navigate into the project directory:

   `cd whatsai`

3. Install the required Python packages:

   `pip install -r requirements.txt`

## Configuration

This app relies on environment variables to configure file paths and processing options. These are loaded using a `.env` file.

### Example `.env` File:

    ORIGINAL_CHAT_FILE=path/to/original_chat.txt
    CLEANED_CHAT_FILE=path/to/cleaned_chat.txt
    CONVERSATION_CHAT_FILE=path/to/conversation_chat.csv
    CLON_NAME=YourBotName
    CONVERSATION_DURATION=4

### Environment Variables:

- `ORIGINAL_CHAT_FILE`: The path to the raw chat log.
- `CLEANED_CHAT_FILE`: The path where the cleaned chat log will be saved.
- `CONVERSATION_CHAT_FILE`: The path where the final CSV with conversation pairs will be saved.
- `CLON_NAME`: The name of the bot or the user whose messages are the "output".
- `CONVERSATION_DURATION`: The duration in hours to separate conversations.

## Training

All the necesary code to train your Bot is in the `training.ipynb` file.
Just run each cell in order.
If you want to play and fine tune your model, you can execute only the require cells. (Ensuring you add the required imports)

> **⚠️ Alert**
>You can download your friends' chat history by following this [guide](https://faq.whatsapp.com/1180414079177245/?locale=et_EE&cms_platform=android)

### Step 1: Clean Chats
- Remove blank lines and unnecessary spaces from chat records.

### Step 2: Create One-Line Conversations
- Format messages as `input, output` and save them in a CSV.

### Step 3: Load the Dataset
- Import the CSV using a library like Pandas. Split the dataset into training and validation sets
- I recommned to test the training importing a subset of data removing the cooment in
```
# Optionally limit to 1000 records for testing
#df = df[:1000]
```

### Step 4: Configure Training
- Define the model architecture, hyperparameters, loss function, and optimizer.

### Step 5: Setup and Train the Model
- Train the model and monitor performance.

### Step 6: Test Text Generation
- Prepare test prompts and evaluate the responses generated by the model.

## Training monitoring
![Screenshot from 2024-10-05 02-25-33](https://github.com/user-attachments/assets/d02375cb-98a8-438d-99de-1040176b366c)

Monitor the training with [tensorboard](https://www.tensorflow.org/tensorboard) executing `tensorboard --logdir OUTPUT_MODEL_DIR/logs` in your terminal

## ChatBot UI
![image](https://github.com/user-attachments/assets/6eb3415e-d723-4b3d-8fba-3cd56cdd55d4)

The Chat UI is built using Streamlit and utilizes the `ChatModel` to facilitate conversations between users and an AI assistant. Below is an overview of its functionality and usage.

### RUN

To start the application, navigate to the root directory of your project and run the following command:

```bash
streamlit run chat_app.py
```

## Project Structure

    ├── README.md
    ├── .env                        # Environment variables
    ├── training.ipynb              # Script for traing your model
    ├── chat_model.py               # Class that manage text generation using your trained model
    ├── chat_app.py                 # Chat interface using streamlit
    ├── requirements.txt            # Python dependencies
    └── input/                      # Directory for storing raw files
    └── output/                     # Directory for storing processed files/models

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.


## License

This project is licensed under the MIT License.
