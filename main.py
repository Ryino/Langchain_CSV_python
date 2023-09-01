# from IPython.display import HTML, display

# def set_css():
#   display(HTML('''
#   <style>
#     pre {
#         white-space: pre-wrap;
#     }
#   </style>
#   '''))
# get_ipython().events.register('pre_run_cell', set_css)

# %pip install --upgrade openai

import openai

# !pip install -q langchain openai chromadb tiktoken

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.callbacks import get_openai_callback
import pandas as pd
import os
import sys
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"]


loader = CSVLoader(file_path='100-SKUs.csv')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

query=input("Any question?\n")
with get_openai_callback() as cb:
    response=chain({"question":query})
    print(response)
    print('-------------------------------------------------------')
    print(cb)
    print('-------------------------------------------------------')

