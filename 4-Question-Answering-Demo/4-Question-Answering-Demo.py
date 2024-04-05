# Importing necessary libraries
import torch
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
)
from scipy.special import softmax
import plotly.express as px
import pandas as pd
import numpy as np

# Defining the context and the question
context = "The giraffe is a large African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes were thought to be one species, Giraffa camelopardalis, with nine subspecies. Most recently, researchers proposed dividing them into up to eight extant species due to new research into their mitochondrial and nuclear DNA, as well as morphological measurements. Seven other extinct species of Giraffa are known from the fossil record."
question = "How many giraffe species are there?"


# Defining the model name and loading the tokenizer and the model
model_name = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


# Tokenizing the context and the question
inputs = tokenizer(question, context, return_tensors="pt")
tokenizer.tokenize(context)


# Running the model and getting the start and end scores
with torch.no_grad():
    outputs = model(**inputs)
start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]



# Creating a dataframe with the scores and plotting them
scores_df = pd.DataFrame({
    "Token Position": list(range(len(start_scores))) * 2,
    "Score": list(start_scores) + list(end_scores),
    "Score Type": ["Start"] * len(start_scores) + ["End"] * len(end_scores),
})
px.bar(scores_df, x="Token Position", y="Score", color="Score Type", barmode="group", title="Start and End Scores for Tokens")



# Getting the answer from the model
start_idx = np.argmax(start_scores)
end_idx = np.argmax(end_scores)
answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
answer = tokenizer.convert_tokens_to_string(answer_tokens)



# Part 2
# Defining a function to predict the answer to a question given a context
def predict_answer(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
    start_idx = np.argmax(start_scores)
    end_idx = np.argmax(end_scores)
    confidence_score = (start_scores[start_idx] + end_scores[end_idx]) /2
    answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer != tokenizer.cls_token:
        return answer, confidence_score
    return None, confidence_score



# Defining a new context and predicting answers to some questions
context = """Coffee is a beverage prepared from roasted coffee beans. Darkly colored, bitter, and slightly acidic, coffee has a stimulating effect on humans, primarily due to its caffeine content. It has the highest sales in the world market for hot drinks.[2][unreliable source?]
...
"""
len(tokenizer.tokenize(context))
predict_answer(context, "What is coffee?")
predict_answer(context, "What are the most common coffee beans?")
predict_answer(context, "How can I make ice coffee?")
predict_answer(context[4000:], "How many people are dependent on coffee for their income?")



# Defining a function to chunk sentences
def chunk_sentences(sentences, chunk_size, stride):
    chunks = []
    num_sentences = len(sentences)
    for i in range(0, num_sentences, chunk_size - stride):
        chunk = sentences[i: i + chunk_size]
        chunks.append(chunk)
    return chunks