# `Auto-evaluator` :brain: :memo:

! See the HuggingFace space for this app: https://huggingface.co/spaces/rlancemartin/auto-evaluator
! See the free-to-use hosted app if you don't have GPT-4 or Anthropic access: https://autoevaluator.langchain.com/
! Code for the hosted app is also open source: https://github.com/langchain-ai/auto-evaluator

This is a lightweight evaluation tool for question-answering using Langchain to:

- Ask the user to input a set of documents of interest

- Apply an LLM (`GPT-3.5-turbo`) to auto-generate `question`-`answer` pairs from these docs

- Generate a question-answering chain with a specified set of UI-chosen configurations

- Use the chain to generate a response to each `question`

- Use an LLM (`GPT-3.5-turbo`) to score the response relative to the `answer`

- Explore scoring across various chain configurations

**Run as Streamlit app**

`pip install -r requirements.txt`

`streamlit run auto-evaluator.py`

**Inputs**

`num_eval_questions` - Number of questions to auto-generate (if the user does not supply an eval set)

`split_method` - Method for text splitting

`chunk_chars` - Chunk size for text splitting
 
`overlap` - Chunk overlap for text splitting
  
`embeddings` - Embedding method for chunks
 
`retriever_type` - Chunk retrieval method

`num_neighbors` - Neighbors for retrieval 

`model` - LLM for summarization of retrieved chunks 

`grade_prompt` - Prompt choice for model self-grading

**Blog**

https://blog.langchain.dev/auto-eval-of-question-answering-tasks/

**UI**

![image](https://user-images.githubusercontent.com/122662504/233218347-de10cf41-6230-47a7-aa9e-8ab01673b87a.png)

**Disclaimer**

```You will need an OpenAI API key with access to `GPT-4` and an Anthropic API key to take advantage of all of the default dashboard model settings. However, additional models (e.g., from Hugging Face) can be easily added to the app.```
