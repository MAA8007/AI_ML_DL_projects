

# Import the necessary modules
import os
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

def get_reply(utterance):
    # Tokenize the utterance
    inputs = tokenizer(utterance, return_tensors="pt")
    # Generate a reply
    reply_ids = model.generate(**inputs)
    # Decode the reply
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Function to clear the terminal window
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Interact with the chatbot in a loop
print("Chatbot is ready. Type 'exit' to end the chat. Type 'clear' to clear the terminal.")
while True:
    # Get user input
    user_input = input("You: ")
    # End the chat if the user types 'exit'
    if user_input.lower() == 'exit':
        break
    # Clear the terminal if the user types 'clear'
    elif user_input.lower() == 'clear':
        clear_terminal()
        continue
    # Get the chatbot's reply
    reply = get_reply(user_input)
    # Print the chatbot's reply
    print(f"Chatbot: {reply}")
