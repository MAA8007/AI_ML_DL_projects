import os
import smtplib
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

def send_email(subject, message, to_email):
    # Email server configuration
    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login('your_email@example.com', 'your_password')
    
    # Send email
    server.sendmail('your_email@example.com', to_email, f"Subject: {subject}\n\n{message}")
    server.quit()

# Interact with the chatbot in a loop
print("Virtual Receptionist is ready. Type 'exit' to end.")
while True:
    # Get user input
    user_input = input("Visitor: ")
    
    # End the chat if user types 'exit'
    if user_input.lower() == 'exit':
        break

    # Custom replies for FAQs or greetings
    lower_input = user_input.lower()
    if "hello" in lower_input or "hi" in lower_input:
        print("Receptionist: Hello! How can I assist you today?")
    elif "services" in lower_input:
        print("Receptionist: We offer a range of services including [list your services here].")
    elif "leave a message" in lower_input:
        subject = input("Please enter the subject of your message: ")
        message = input("Please enter your message: ")
        department = input("Which department would you like to send this message to? ")
        email_map = {"sales": "sales@example.com", "support": "support@example.com"} # Add mapping here
        send_email(subject, message, email_map.get(department, "default@example.com"))
        print("Receptionist: Your message has been forwarded to the relevant department.")
    else:
        # Get the chatbot's reply
        reply = get_reply(user_input)
        print(f"Receptionist: {reply}")
