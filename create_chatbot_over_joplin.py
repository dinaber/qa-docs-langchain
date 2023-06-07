"""
Create a CHATGPT chatbot that can perform Q&A over Joplin notes.
Created on Jun 2023

@author: Dina Berenbaum
"""

import os
import glob

import weaviate

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate

from mysystemfuncs import get_key_from_config


def set_env_vars():
    os.environ["OPENAI_API_TYPE"] = get_key_from_config('OPENAI_API_TYPE')
    os.environ["OPENAI_API_VERSION"] = get_key_from_config('OPENAI_API_VERSION')
    os.environ["OPENAI_API_BASE"] = get_key_from_config('OPENAI_API_BASE')
    os.environ["OPENAI_API_KEY"] = get_key_from_config('OPENAI_API_KEY')


set_env_vars()


class JoplinChatbot:
    def __init__(self, sandbox_url, index_name='Joplin_test',
                 embedder=OpenAIEmbeddings,
                 splitter=RecursiveCharacterTextSplitter,
                 vectorstore=Weaviate, **kwargs):
        """
        Initializer for the JoplinChatbot class. Vectorestore and Embedder are needed both for adding new files to the
        datastore and also for the queries. The splitter is only needed for adding new files to the datastore.

        :param sandbox_url: URL for the sandbox.
        :param index_name: Index name for the chatbot.
        :param embedder: Class for the embedder (defaults to OpenAIEmbeddings).
        :param splitter: Class for the splitter (defaults to RecursiveCharacterTextSplitter).
        :param vectorstore: Class for the vectorstore (defaults to Weaviate).
        :param splitter_args: Additional arguments for the splitter class (optional).
        :param embedder_args: Additional arguments for the embedder class (optional).
        :param chain_type: Type of chain to use (defaults to "stuff").
        """
        set_env_vars()
        self._index_name = index_name
        self._client = weaviate.Client(sandbox_url)

        # Chosen classes
        self._embedder_class = embedder
        self._splitter_class = splitter
        self._vectorstore_class = vectorstore

        # Initialize the modules
        self._embedding = self._get_embedding(kwargs.get("embedder_params", {}))
        self._vectorstore = self._get_vectorstore(kwargs.get("vectorstore_params", {}))
        self.db = VectorStoreIndexWrapper(vectorstore=self._vectorstore)

        self.chain_type = kwargs.get("chain_type", "stuff")  # 4 types of chains: stuff, map_reduce, refine, map_rerank

    def _read_markdown_files(self, path_to_dir):
        """
        Reads markdown files from a directory.

        :param path_to_dir: The directory to read markdown files from.
        :return: List of loaded markdown files.
        """
        markdown_files = glob.glob(os.path.join(path_to_dir, "*.md"))
        return [UnstructuredMarkdownLoader(f).load()[0] for f in markdown_files]

    def _get_splitter(self, splitter_args):
        """
        Returns a splitter instance.

        :param splitter_args: Arguments for the splitter class.
        :return: An instance of a splitter.
        """
        return self._splitter_class(**splitter_args)

    def _get_embedding(self, embedder_args):
        """
        Returns an embedder instance.

        :param embedder_args: Arguments for the embedder class.
        :return: An instance of an embedder.
        """
        return self._embedder_class(**embedder_args)

    def _get_vectorstore(self, vectorstore_args):
        """
        Returns a vectorstore instance.

        :return: An instance of a vectorstore.
        """
        return self._vectorstore_class(self._client, index_name=self._index_name, text_key='text',
                                       embedding=self._embedding,
                                       by_text=False, **vectorstore_args)

    def index_new_files(self, path_to_dir, splitter_args=None):
        """
        Indexes new files.

        :param path_to_dir: Directory containing the new files.
        :param splitter_args: Additional arguments for the splitter (optional).
        """
        documents = self._read_markdown_files(path_to_dir)
        splitter = self._get_splitter(splitter_args if splitter_args else {})

        # currently for the from_text method of Weaviate you need to pass the
        # arguments outside of the initialized vectorestore, so it means you pass them twice :/
        vectorstore_kwargs = {"client": self._client, "index_name": self._index_name}

        self.db = VectorstoreIndexCreator(embedding=self._embedding, vectorstore_cls=self._vectorstore_class,
                                          text_splitter=splitter, vectorstore_kwargs=vectorstore_kwargs).from_documents(
            documents)

    def query(self, text):
        """
        Queries the database.

        :param text: The query text.
        :return: The result of the query.
        """
        deployment_name = get_key_from_config('OPENAI_DEPLOYMENT')  # todo: make sure this deployment is really running
        llm = AzureOpenAI(deployment_name=deployment_name)
        return self.db.query(text, chain_type=self.chain_type, llm=llm)
