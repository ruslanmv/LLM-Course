**Semantic Indexing: Unlocking Meaningful Search for Large Language Models**

In the realm of Large Language Models (LLMs), where vast amounts of text reside, efficiently retrieving relevant information becomes paramount. Keyword-based search often falls short, as it struggles to grasp the underlying meaning and context of queries. This is where **semantic indexing** steps in, acting as a powerful bridge between LLMs and the intricate world of human language.

**What is Semantic Indexing?**

Semantic indexing is a technique that goes beyond mere keyword matching. It leverages the power of **sentence transformers**, specialized neural networks adept at capturing the semantic relationships between words and sentences. By converting text into numerical representations known as **embeddings**, semantic indexing allows LLMs to understand the essence of a query and its connection to stored information.

**Delving into the Code**

The provided Python code demonstrates a practical implementation of semantic indexing with LLMs. Let's break it down step-by-step:

```python
# Import required libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
```

- We begin by importing essential libraries:
    - `load_dataset` from `datasets`: Facilitates loading datasets from Hugging Face Hub.
    - `SentenceTransformer` and `util` from `sentence_transformers`: Provide tools for sentence transformers and utility functions.
    - `torch`: The fundamental deep learning framework used by the sentence transformer model.

```python
# Load the multi_news dataset from Hugging Face and take only the 'test' split for efficiency
dataset = load_dataset("multi_news", split="test")
```

- We fetch the "multi_news" dataset from Hugging Face Hub, containing a variety of news articles. Here, we only consider the test split for demonstration purposes (you might use the entire dataset for real-world applications).

```python
# Convert the test dataset to a pandas DataFrame and take only 2000 random samples
df = dataset.to_pandas().sample(2000, random_state=42)
```

- The loaded dataset is transformed into a pandas DataFrame for easier manipulation. We then extract a random subset of 2000 samples to enhance efficiency.

```python
# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
```

- This line is crucial. We load a pre-trained sentence transformer model, "all-MiniLM-L6-v2" in this case. This model has been trained on a massive corpus of text, empowering it to grasp the nuances of language. You can explore other pre-trained models available from Sentence Transformers ([https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)).

```python
# Encode each summary in the DataFrame using the sentence transformer model and store the embeddings in a list
passage_embeddings = list(model.encode(df['summary'].to_list(), show_progress_bar=True))

# Print the shape of the first passage embedding
passage_embeddings[0].shape
```

- Here's the core of semantic indexing. We employ the sentence transformer model (`model.encode()`) to encode each news article's summary (stored in the 'summary' column of the DataFrame) into a numerical embedding. These embeddings capture the semantic meaning of the summaries. Finally, we store these embeddings in a list named `passage_embeddings`.

```python
# Declare a query string
query = "Find me some articles about technology and artificial intelligence"

# Define a function to find relevant news articles based on a given query
def find_relevant_news(query):
  # Encode the query using the sentence transformer model
  query_embedding = model.encode(query)
  # Print the shape of the query embedding
  query_embedding.shape

  # Calculate the cosine similarity between the query embedding and the passage embeddings
  similarities = util.cos_sim(query_embedding, passage_embeddings)

  # Find the indices of the top 3 most similar passages
  top_indicies = torch.topk(similarities.flatten(), 3).indices

  # Get the top 3 relevant passages by slicing the summaries at 200 characters and adding an ellipsis
  top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indicies]

  # Return the top 3 relevant passages
  return top_relevant_passages

# Find relevant news articles for different queries
find_relevant_news("Natural disasters")
find_



Absolutely! Here's the continuation of the blog section on Semantic Indexing for LLMs, picking up from where you requested (returning the top 3 relevant passages):

**# Return the Top 3 Relevant Passages**

Now, let's delve into the heart of information retrieval:

```python
  # Return the top 3 relevant passages
  return top_relevant_passages
```

This line marks the culmination of the search process. We've calculated the cosine similarities between the query embedding and each passage embedding, essentially measuring how semantically close the query is to each news article summary.

**Cosine Similarity: Unveiling Semantic Closeness**

Cosine similarity is a metric that quantifies the similarity between two vectors. In our case, the vectors are the query embedding and the passage embeddings. A higher cosine similarity value indicates greater semantic closeness.

**Finding the Top Matches**

- `torch.topk(similarities.flatten(), 3).indices`: This line identifies the indices of the top 3 most similar passages based on the calculated cosine similarities. The `flatten()` method ensures we're working with a one-dimensional tensor, and `topk` retrieves the top k (in this case, 3) elements along with their corresponding indices.

**Retrieving the Top Articles**

- We leverage these top indices to extract the actual summaries from the DataFrame (`df`). Slicing each summary at 200 characters and appending an ellipsis ("...") ensures the returned passages are concise while providing informative snippets.

**Putting It All Together: Function in Action**

The `find_relevant_news` function encapsulates the entire search process. When invoked with a query string (e.g., "Natural disasters"), it retrieves and returns the top 3 most relevant news article summaries based on their semantic similarity to the query.

```python
# Find relevant news articles for different queries
find_relevant_news("Natural disasters")
find_relevant_news("Law enforcement and police")
find_relevant_news("Politics, diplomacy and nationalism")
```

These lines demonstrate how to utilize the `find_relevant_news` function for various queries. The returned top 3 relevant passages will provide summaries that are semantically aligned with the respective queries, showcasing the power of semantic indexing in LLM-based search.

**In Conclusion**

Semantic indexing, powered by sentence transformers, empowers LLMs to transcend keyword matching and delve into the true meaning of text. It enables LLMs to understand the essence of queries and efficiently retrieve relevant information, paving the way for more meaningful and insightful interactions between humans and large language models.