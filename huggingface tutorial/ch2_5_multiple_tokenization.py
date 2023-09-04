from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Initialize the tokenizer and the model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

# Single sequence example
single_sequence = "I love this movie."

# Multiple sequence example
multiple_sequences = ["I love this movie.", "This is the best film I've ever seen."]

# Tokenize the sequences
single_sequence_tokenized = tokenizer(single_sequence, return_tensors="tf", padding='longest', truncation=True)
multiple_sequences_tokenized = tokenizer(multiple_sequences, return_tensors="tf", padding='longest', truncation=True)

# Note the shape difference
print(f"Single sequence shape: {single_sequence_tokenized['input_ids'].shape}")
print(f"Multiple sequences shape: {multiple_sequences_tokenized['input_ids'].shape}")

# Make predictions
single_sequence_prediction = model(single_sequence_tokenized)
multiple_sequences_predictions = model(multiple_sequences_tokenized)

# Display the logits (raw prediction values) for each sequence
print(f"Single sequence prediction: {single_sequence_prediction.logits}")
print(f"Multiple sequences predictions: {multiple_sequences_predictions.logits}")

predictions = tf.math.softmax(multiple_sequences_predictions.logits, axis=-1)
print(predictions)



#Padding: Padding is the process of adding special tokens (usually zeros) to the end of sequences to make them of equal length. In NLP, sentences or documents often have different lengths, but many models require fixed-size input tensors. Padding ensures that all sequences have the same length, enabling efficient batch processing. The added padding tokens typically don't carry any meaningful information and are ignored by the model during computation.

#Truncation: Truncation involves removing tokens from sequences that exceed a specified maximum length. In cases where input sequences are too long to fit within the model's capacity or computational constraints, truncation can be used to discard the excess tokens from the end of the sequence. This way, the sequence length is reduced to a manageable size, although some information at the end of the sequence may be lost.

#Attention Mask: Attention mechanisms play a crucial role in transformer models, allowing them to focus on relevant parts of the input sequence during computation. An attention mask is a binary mask that indicates which elements of the input should be attended to and which should be ignored. In the context of variable-length sequences, an attention mask is used to differentiate between actual tokens and padded tokens. By masking out the padded tokens, the model can avoid assigning importance or attending to them, preventing them from affecting the model's attention or computation.