import os
import json
import time
import pypdf
import random
import itertools
import text_utils
import pandas as pd
import altair as alt
import streamlit as st
from io import StringIO
from langchain.llms import Anthropic
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.evaluation.qa import QAEvalChain
from langchain.retrievers import TFIDFRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

if "existing_df" not in st.session_state:
    summary = pd.DataFrame(columns=['chunk_chars',
                                    'overlap',
                                    'split',
                                    'model',
                                    'retriever',
                                    'embedding',
                                    'latency',
                                    'retrival score',
                                    'answer score'])
    st.session_state.existing_df = summary
else:
    summary = st.session_state.existing_df


@st.cache_data
def load_docs(files):

    # Load docs
    # IN: List of upload files (from Streamlit)
    # OUT: str
    # TODO: Support multple docs, Use Langchain loader

    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = pypdf.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text = text_utils.clean_pdf_text(text)
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")

    return all_text


@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: list of JSON
    # TODO: Refactor, PR for Langchain repo, add error handling for JSON load in QA chain

    st.info("`Generating eval set ... I will generate question / answer pairs from the input text. If you want to grade specific question - answer pairs, just input a JSON above`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
        except:
            st.warning('Error generating question %s.'%str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split text
    # IN: text, chunk size, overlap
    # OUT: list of str splits
    # TODO: Add parameter for splitter type

    st.info("`Splitting and embedding doc ...`")
    if split_method == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(separator=" ",
                                              chunk_size=chunk_size,
                                              chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    return splits


@st.cache_resource
def make_retriever(splits, retriever_type, embeddings, num_neighbors):

    # Make retriever
    # IN: list of str splits, retriever type, and embedding type
    # OUT: retriever

    if retriever_type == "similarity-search":
        if embeddings == "OpenAI":
            try:
                vectorstore = FAISS.from_texts(splits, OpenAIEmbeddings())
            except ValueError:
                st.warning("`Error using OpenAI embeddings (disallowed TikToken token in the text). Using HuggingFace.`", icon="⚠️")
                vectorstore = FAISS.from_texts(splits, HuggingFaceEmbeddings())
        elif embeddings == "HuggingFace":
            vectorstore = FAISS.from_texts(splits, HuggingFaceEmbeddings())
        retriever = vectorstore.as_retriever(k=num_neighbors)
    elif retriever_type == "TF-IDF":
        retriever = TFIDFRetriever.from_texts(splits,k=num_neighbors)
    return retriever


def make_chain(model_version, retriever):

    # Make chain
    # IN: model version, vectorstore
    # OUT: chain
    # TODO: Support for multiple model types

    if (model_version == "gpt-3.5-turbo") or (model_version == "gpt-4"):
        llm = ChatOpenAI(model_name=model_version, temperature=0)
    
    elif model_version == "anthropic":
        llm = Anthropic(temperature=0)
    qa = RetrievalQA.from_chain_type(llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     input_key="question")
    return qa


def grade_model_answer(predicted_dataset, predictions):

    # Grade the model
    # IN: model predictions and ground truth (lists)
    # OUT: list of scores
    # TODO: Support for multiple grading types

    st.info("`Grading model answer ...`")
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), 
                                      prompt=GRADE_ANSWER_PROMPT)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs

def grade_model_retrieval(gt_dataset, predictions):
    
    # Grade the docs retrieval
    # IN: model predictions and ground truth (lists)
    # OUT: list of scores
    # TODO: Support for multiple grading types

    st.info("`Grading relevance of retrived docs ...`")
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), 
                                      prompt=GRADE_DOCS_PROMPT)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs

def run_eval(chain, retriever, eval_set):

    # Compute eval
    # IN: chain and ground truth (lists)
    # OUT: list of scores, latenct, predictions
    # TODO: Support for multiple grading types

    st.info("`Running eval ...`")
    predictions = []
    retrived_docs = []
    gt_dataset = []
    latency = []
    
    for data in eval_set:
        
        # Get answer and log latency
        start_time = time.time()
        predictions.append(chain(data))
        gt_dataset.append(data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        latency.append(elapsed_time)
        
        # Retrive data
        docs=retriever.get_relevant_documents(data["question"])
        
        # Extract text from retrived docs
        retrived_doc_text = ""
        for i,doc in enumerate(docs):
            retrived_doc_text += "Doc %s: "%str(i+1) + doc.page_content + " "
        retrived = {"question": data["question"],"answer": data["answer"], "result": retrived_doc_text}
        retrived_docs.append(retrived)
        
    # Grade
    graded_answers = grade_model_answer(gt_dataset, predictions)
    graded_retrieval = grade_model_retrieval(gt_dataset, retrived_docs)
    return graded_answers, graded_retrieval, latency, predictions

# Auth
st.sidebar.image("img/diagnostic.jpg")

num_eval_questions = st.sidebar.select_slider("`Number of eval questions`",
                                      options=[1, 5, 10, 15, 20], value=5)

chunk_chars = st.sidebar.select_slider("`Choose chunk size for splitting`",
                               options=[500, 750, 1000, 1500, 2000], value=1000)

overlap = st.sidebar.select_slider("`Choose overlap for splitting`",
                           options=[0, 50, 100, 150, 200], value=100)

split_method = st.sidebar.radio("`Split method`",
                                ("RecursiveTextSplitter",
                                 "CharacterTextSplitter"),
                                index=0)

model = st.sidebar.radio("`Choose model`",
                         ("gpt-3.5-turbo",
                          "gpt-4",
                          "anthropic"),
                         index=0)

retriever_type = st.sidebar.radio("`Choose retriever`",
                                 ("TF-IDF",
                                  "similarity-search"),
                                 index=1)

num_neighbors = st.sidebar.select_slider("`Choose # chunks to retrieve`",
                                 options=[3, 4, 5, 6, 7, 8])

embeddings = st.sidebar.radio("`Choose embeddings`",
                              ("HuggingFace",
                               "OpenAI"),
                              index=1)

# App
st.header("`Auto-evaluator`")
st.info("`I am an evaluation tool for question-answering. Given documents, I will auto-generate a question-answer eval set and evaluate using the selected chain settings. Experiments with different configurations are logged. Optionally, provide your own eval set.`")
uploaded_eval_set = st.file_uploader("`[Optional] Please upload eval set (JSON):` ",
                                     type=['json'],
                                     accept_multiple_files=False)

uploaded_file = st.file_uploader("`Please upload a file to evaluate (.txt or .pdf):` ",
                                 type=['pdf', 'txt'],
                                 accept_multiple_files=True)

if uploaded_file:

    # Load docs
    text = load_docs(uploaded_file)
    # Generate num_eval_questions questions, each from context of 3k chars randomly selected
    if not uploaded_eval_set:
        eval_set = generate_eval(text, num_eval_questions, 3000)
    else:
        eval_set = json.loads(uploaded_eval_set.read())
    # Split text
    splits = split_texts(text, chunk_chars, overlap, split_method)
    # Make vector DB
    retriever = make_retriever(splits, retriever_type, embeddings, num_neighbors)
    # Make chain
    qa_chain = make_chain(model, retriever)
    # Grade model
    graded_answers, graded_retrieval, latency, predictions = run_eval(qa_chain, retriever, eval_set)
    
    # Assemble ouputs
    d = pd.DataFrame(predictions)
    d['answer score'] = [g['text'] for g in graded_answers]
    d['docs score'] = [g['text'] for g in graded_retrieval]
    d['latency'] = latency

    # Summary statistics
    mean_latency = d['latency'].mean()
    correct_answer_count = len([text for text in d['answer score'] if text == "CORRECT"])
    correct_docs_count = len([text for text in d['docs score'] if "Context is relevant: True." in text])
    percentage_answer = (correct_answer_count / len(graded_answers)) * 100
    percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100
    
    st.subheader("`Run Results`")
    st.info("`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for grading in text_utils`")
    st.dataframe(data=d, use_container_width=True)
    
    # Accumulate results
    st.subheader("`Aggregate Results`")
    new_row = pd.DataFrame({'chunk_chars': [chunk_chars],
                            'overlap': [overlap],
                            'split': [split_method],
                            'model': [model],
                            'retriever': [retriever_type],
                            'embedding': [embeddings],
                            'latency': [mean_latency],
                            'retrival score': [percentage_docs],
                            'answer score': [percentage_answer]})
    summary = pd.concat([summary, new_row], ignore_index=True)
    st.dataframe(data=summary, use_container_width=True)
    st.session_state.existing_df = summary
    
    # Dataframe for visualization
    show = summary.reset_index().copy()
    show.columns = ['expt number', 'chunk_chars', 'overlap',
                    'split', 'model', 'retriever', 'embedding', 'latency', 'retrival score','answer score']
    show['expt number'] = show['expt number'].apply(
        lambda x: "Expt #: " + str(x+1))
    show['mean score'] = (show['retrival score'] + show['answer score']) / 2
    c = alt.Chart(show).mark_circle().encode(x='mean score', y='latency',
                                             color='expt number', tooltip=['expt number', 'mean score', 'latency'])
    st.altair_chart(c, use_container_width=True, theme="streamlit")
