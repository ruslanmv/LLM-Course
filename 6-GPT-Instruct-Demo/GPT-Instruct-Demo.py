from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset



dataset = load_dataset("hakurei/open-instruct-v1", split='train')
dataset.to_pandas().sample(20)



def preprocess(example):
    example['prompt'] = f"{example['instruction']} {example['input']} {example['output']}"

    return example

def tokenize_datasets(dataset):
    tokenized_dataset = dataset.map(lambda example: tokenizer(example['prompt'], truncation=True, max_length=128), batched=True, remove_columns=['prompt'])

    return tokenized_dataset




dataset = dataset.map(preprocess, remove_columns=['instruction', 'input', 'output'])
dataset =  dataset.shuffle(42).select(range(100000)).train_test_split(test_size=0.1, seed=42)



train_dataset = dataset['train']
test_dataset = dataset['test']



MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = tokenize_datasets(train_dataset)
test_dataset = tokenize_datasets(test_dataset)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)




data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

traing_args = TrainingArguments(output_dir="models/diablo_gpt",
                                num_train_epochs=1,
                                per_device_train_batch_size=32,
                                per_device_eval_batch_size=32)\

trainer = Trainer(model=model,
                    args=traing_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    data_collator=data_collator)



trainer.train()



# Get the trained checkpoint directly
model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct")



def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
    outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated[:generated.rfind('.')+1]



generate_text("What's the best way to cook chiken breast?")

generate_text("Should I invest stocks?")

generate_text("I need a place to go for this summer vacation, what locations would you recommend")

generate_text("What's the fastest route from NY City to Boston?")