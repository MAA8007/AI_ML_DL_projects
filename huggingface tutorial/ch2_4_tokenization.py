#tutorial 4 and 5 describe tokenizers in depth 

from transformers import AutoTokenizer
import tensorflow as tf


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#or, we can also use:

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Translating text to numbers is known as encoding. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.
#As we’ve seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called tokens. There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.

sequence = "I've been waiting for a HuggingFace course my whole life. its great"

tokens = tokenizer.tokenize(sequence)
#['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
ids = tokenizer.convert_tokens_to_ids(tokens)
#[7993, 170, 11303, 1200, 2443, 1110, 3014]
input_ids = tf.constant(ids)









#<--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->








#Decoding is going the other way around: from vocabulary indices, we want to get a string. This can be done with the decode() method as follows:
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
#'Using a Transformer network is simple'