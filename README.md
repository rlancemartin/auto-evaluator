# `Auto-evaluator`

This is a lightweight evaluation tool for question-answering using Langchain to:

* Auto-generate an eval set given a doc (or set of docs); optionally, you can also supply your own eval set
* Evaluate a QA chain with specified QA chain configurations (model, embeddings, retriver, etc)
* The prompts used for evluation are specified in `text_utils` and can be easily tuned / engineered
* Test different configurations and log results from each experiments 

**Run as Streamlit app**

Note: you will need an OpenAI API key with with access to `GPT-4` and an Anthropic API key to take advantage of all of the default dashboard model settings. However, additional models (e.g., from HuggingFace) can be easily added to the app.

`pip install -r requirements.txt`
`streamlit run auto-evaluator.py`
