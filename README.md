# Chatbot-

Open the Terminal & Create a Virtual Environment  >>>>GIt bash

>>python -m venv venv


>>source venv/Scripts/activate



#📌Step 1: Install Dependencies
Open VS Code, and install the required Python libraries using the terminal:

>>>pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers ctransformers pypdf asyncio


#📌Step 4: Download the Model
Go to TheBloke’s HuggingFace LLaMA models.
Download llama-2-7b-chat.ggmlv3.q4_0.bin (or a smaller version if needed).
Create a folder named model/ in your VS Code project directory.
Move the downloaded .bin file into model/.


#📌 Step 5: Prepare Your Data
Create a data/ folder inside your project directory.
Add any PDF files you want to process into data/.



#📌 Step 6: Run the Chatbot
Open VS Code terminal (Ctrl+ ` or View > Terminal).

Run the chatbot:
>> python chatbot.py
