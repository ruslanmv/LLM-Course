#!pip install numpy==1.25.1
#!pip install transformers
#!pip install datasets===2.13.1

# Importing necessary modules
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# Loading the dataset and removing unnecessary columns
dataset = load_dataset('amazon_us_reviews', 'Electronics_v1_00', split='train')
dataset = dataset.remove_columns([x for x in dataset.features if x not in ['review_body', 'verified_purchase', 'review_headline', 'product_title', 'star_rating']])

# Filtering the dataset and encoding the 'star_rating' column
dataset = dataset.filter(lambda x: x['verified_purchase'] and len(x['review_body']) > 100).shuffle(42).select(range(100000))
dataset = dataset.class_encode_column("star_rating")

# Splitting the dataset into training and testing sets
dataset = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="star_rating")

train_dataset = dataset['train']
test_dataset = dataset['test']


# Initializing the tokenizer
MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained('t5-base')|


# Defining the function to preprocess the data
def preprocess_data(examples):
    examples['prompt'] = [f"review: {example['product_title']}, {example['star_rating']} Stars!" for example in examples]
    examples['response'] = [f"{example['review_headline']} {example['review_body']}" for example in examples]

    inputs = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=128)
    targets = tokenizer(examples['response'], padding='max_length', truncation=True, max_length=128)

    # Set -100 at the padding positions of target tokens
    target_input_ids = []
    for ids in targets['input_ids']:
        target_input_ids.append([id if id != tokenizer.pad_token_id else -100 for id in ids])

    inputs.update({'labels': target_input_ids})
    return inputs

# Preprocessing the training and testing datasets
train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Fine-tuning the T5 model
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

TRAINING_OUTPUT = "./models/t5_fine_tuned_reviews"
training_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()


# Saving the model
trainer.save_model(TRAINING_OUTPUT)


# Loading the fine-tuned model
model = T5ForConditionalGeneration.from_pretrained(TRAINING_OUTPUT)

# or get it directly trained from here:
# model = T5ForConditionalGeneration.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")


# Defining the function to generate reviews
def generate_review(text):
    inputs = tokenizer("review: " + text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=128, no_repeat_ngram_size=3, num_beams=6, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Generating reviews for random products
random_products = test_dataset.shuffle(42).select(range(10))['product_title']

print(generate_review(random_products[0] + ", 3 Stars!"))
print(generate_review(random_products[1] + ", 5 Stars!"))