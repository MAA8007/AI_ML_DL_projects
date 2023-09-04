from flask import Flask, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)

# Setup the model and tokenizer
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

def should_escalate(query):
    # Add your logic here to determine if the query should be escalated to a human agent
    # You can use various conditions or thresholds based on your specific requirements
    # For example, you might check for certain keywords or phrases that indicate a need for human intervention
    # Return True if the query should be escalated, False otherwise
    return False

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json['message']
    
    # Check if the query should be escalated
    if should_escalate(message):
        reply = "I'm sorry, but I am unable to assist with that. Please wait while I connect you to a human agent."
        # Add code here to handle the escalation process (e.g., send an email notification to the support team)
    else:
        reply = get_reply(message)
    
    return jsonify({'reply': reply})

# Run Flask app
if __name__ == "__main__":
    app.run()
