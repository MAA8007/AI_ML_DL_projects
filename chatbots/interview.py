import os
import requests
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration


# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Download the necessary NLTK resources

def get_reply(utterance, language='en'):
    # Tokenize the utterance
    inputs = tokenizer(utterance, return_tensors="pt")
    # Generate a reply
    reply_ids = model.generate(**inputs)
    # Decode the reply
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

def correct_grammar(text, language='en'):
    url = "https://api.languagetoolplus.com/v2/check"
    headers = {"content-type": "application/x-www-form-urlencoded"}
    data = f"text={text}&language={language}"
    response = requests.post(url, data=data, headers=headers)
    corrections = response.json()
    return corrections

def display_grammar_feedback(corrections):
    matches = corrections.get('matches', [])
    if matches:
        print("\nGrammar feedback:")
        for match in matches:
            message = match.get('message', 'N/A')
            suggestions = ", ".join([rep.get('value', '') for rep in match.get('replacements', [])])
            print(f"- {message}")
            if suggestions:
                print(f"  Suggested corrections: {suggestions}\n")
    else:
        print("No grammar issues found.")

# Interact in a loop
print("Educational Assistant is ready. Type 'exit' to end.")
language = input("Choose a language (e.g., en, fr, de): ")

while True:
    # Get user input
    user_input = input("You: ")
    
    # End the chat if user types 'exit'
    if user_input.lower() == 'exit':
        break

    # Get the chatbot's reply
    reply = get_reply(user_input, language)
    print(f"Assistant: {reply}")
    
    # Correct grammar
    corrections = correct_grammar(user_input, language)
    display_grammar_feedback(corrections)
