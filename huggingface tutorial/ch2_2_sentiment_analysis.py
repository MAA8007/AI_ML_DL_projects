from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf


#The tokenizer and model should always be from the same checkpoint.

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I thought this was going to exceed my expectations, but ive been proven wrong",
    "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m "
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

#TFAutoModel only needs to know the checkpoint from which to initialize to return the correct architecture.
outputs = model(inputs)

predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)

#the tokenizer itself is very powerful and ready to use. In the next 2 tutorials, it will be broken down.

