from datasets import load_dataset
from transformers import AutoTokenizer

# Specify the path to your CSV file
data_files = {"train": "translator/deu_train.csv", "test": "translator/deu_test.csv"}

# Load the dataset
raw_datasets = load_dataset("csv", data_files=data_files)

# Function to convert to desired format
def format_translation(example):
    example["translation"] = {"en": example["English"], "de": example["German"]}
    return example

# Apply the format function to train and test datasets
formatted_datasets = {}
formatted_datasets["train"] = raw_datasets["train"].map(format_translation)
formatted_datasets["test"] = raw_datasets["test"].map(format_translation)

model_checkpoint = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="tf")


max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["de"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = {}
tokenized_datasets["train"] = formatted_datasets["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=formatted_datasets["train"].column_names,
)
tokenized_datasets["test"] = formatted_datasets["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=formatted_datasets["test"].column_names,
)


from transformers import TFAutoModelForSeq2SeqLM

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["test"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

from datasets import load_metric

metric = load_metric("sacrebleu")

import numpy as np
import tensorflow as tf
from tqdm import tqdm

generation_data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128
)

tf_generate_dataset = model.prepare_tf_dataset(
    tokenized_datasets["test"],
    collate_fn=generation_data_collator,
    shuffle=False,
    batch_size=8,
)



def generate_with_xla(batch):
    return model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=128,
    )


def compute_metrics():
    all_preds = []
    all_labels = []

    for batch, labels in tqdm(tf_generate_dataset):
        predictions = generate_with_xla(batch)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = labels.numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    result = metric.compute(predictions=all_preds, references=all_labels)
    return {"bleu": result["score"]}

print(compute_metrics())

from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(
    output_dir="en-de-mymodel", tokenizer=tokenizer
)

model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)

print(compute_metrics())

from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "en-de-mymodel"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")