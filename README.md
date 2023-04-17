# `Auto-evaluator` :brain: :memo:

This is a lightweight evaluation tool for question-answering using Langchain to:

- Ask the user to input a set of documents of interest

- Use an LLM (`GPT-3.5-turbo`) to auto-generate `question``answer` pairs from these docs

- Generate a question-answering chain with a specified set of UI-chosen configurations

- Use the chain to generate a response to each `question`

- Use an LLM (`GPT-3.5-turbo`) to score the response relative to the `answer`

- Explore scoring across various chain configurations

**Run as Streamlit app**

`pip install -r requirements.txt`

`streamlit run auto-evaluator.py`

**Inputs**

`num_eval_questions` - Number of question to auto-generate (if the user does not supply an eval set)

`split_method` - Method for text splitting

`chunk_chars` - Chunk size for text splitting
 
`overlap` - Chunk overlap for text splitting
  
`embeddings` - Embedding method for chunks
 
`retriever_type` - Chunk retrival method

`num_neighbors` - Neighbors in retrivial 

`model` - LLM for summarization from retrived chunks 

`grade_prompt` - Promp choice for grading

**Blog**

https://blog.langchain.dev/auto-eval-of-question-answering-tasks/

**UI**

 ![ui](https://user-images.githubusercontent.com/122662504/232509494-3d1777f9-55f1-496b-b102-70543d2bb17f.jpeg)

**Disclaimer**

```You will need an OpenAI API key with with access to `GPT-4` and an Anthropic API key to take advantage of all of the default dashboard model settings. However, additional models (e.g., from HuggingFace) can be easily added to the app.```
