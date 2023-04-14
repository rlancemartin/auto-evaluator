# `Auto-evaluator`

This is a lightweight evaluation tool for question-answering using Langchain to:
* Auto-generate an eval set given a doc (or set of docs)
* Evaluate a QA chain with specified configurations (model, embeddings, etc)
* Log results from experiments 

**Run as Streamlit app**

Note: you will need an OpenAI API key with with access to GPT-4 and an Anthropic API key to take advantage of the default dashboard model settings. 

`pip install -r requirements.txt`

`streamlit run auto-evaluator.py`


