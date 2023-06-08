# QA-Docs-Langchain

Welcome to QA-Docs-Langchain! This repository provides a simple implementation of a code that utilizes the LangChain library alongside Language Models (LLMs) 
to facilitate efficient in-house Question & Answer (Q&A) capabilities for your own documents.
Say goodbye to the hassle of training or fine-tuning your own models and save valuable time and resources. 
Instead, you can effortlessly break down your documents into contextually fitting chunks, embed their semantic meaning into vectors, 
index them in a vector store, and retrieve the most relevant chunks in real-time based on their semantic similarity. 

This is a very basic proof of concept that accompanies the <<<>>> blogpost. For a detailed explanation and technical step, check out the article.

## Table of Contents
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Getting Started

To get started with QA-Docs-Langchain, you can follow these steps:

1. Clone this repository

2. Before running the project, make sure to obtain and set the following enviroment variables:
OPENAI_API_KEY, OPENAI_DEPLOYMENT, OPENAI_BASE, OPENAI_API_TYPE (= azure), OPENAI_API_VERSION, WEAVIATE_JOPLIN_SANDBOX_URL

## Usage

The following examples demonstrate how to use the QA-Docs-Langchain functionality:
For example of params and other functionalities check out **example_qa_chain_go_book.ipynb **notebook.
1. Initialization 

    To initialize the JoplinChatbot class, use the following code snippet in your implementation:
    ```python
      import os
      from qa_docs_langchain import JoplinChatbot

      jchat = JoplinChatbot(os.environ.get("WEAVIATE_JOPLIN_SANDBOX_URL"), **params)

2. Indexing into vectorestores
    ```python
      path_to_dir = "/path/to/directory"
      jchat.index_new_files(path_to_dir, splitter_args=params["splitter_params"])
      
3. Querying
    ```python
      query = "What is the purpose of the Langchain project?"
      result = jchat.query(query)
      
## License      

This repository is licensed under the MIT License. You are free to use, modify, and distribute this codebase as per the terms of the license.
If you have any questions or need further assistance, please don't hesitate to reach out to me.
