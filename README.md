# RAG Model using LangChain and Milvus

## Overview
This project implements a **Retrieval-Augmented Generation** (RAG) model using **LangChain** and **Milvus** as the vector database. The model leverages document retrieval techniques such as **BM25** and **vector search** to enhance responses from a language model hosted on Hugging Face.

## Features
- **Document Processing**: Loads and splits PDFs into manageable text chunks.
- **Embeddings**: Uses Hugging Face's `all-MiniLM-L6-v2` for text embeddings.
- **Vector Storage**: Stores embeddings in a Milvus vector database.
- **Hybrid Retrieval**: Combines BM25 keyword search with vector-based retrieval.
- **LLM Integration**: Uses `google/gemma-2b` hosted on Hugging Face for text generation.
- **Custom Prompting**: Implements a structured prompt template for improved response quality.

## Installation
Ensure you have Python installed, then install dependencies:
```bash
pip install langchain_milvus langchain_community pypdf huggingface_hub rank_bm25 pymilvus
```

## Usage
### 1. Set up environment variables
Export your Hugging Face API key:
```bash
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

### 2. Load and Process Documents
Load a PDF and split it into text chunks:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("path_to_pdf.pdf")
pages = loader.load_and_split()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)
```

### 3. Initialize Milvus Vector Database
```python
from pymilvus import MilvusClient
from langchain_milvus import Milvus

MILVUS_URL = "./hybrid_search.db"
client = MilvusClient(uri=MILVUS_URL)

if client.has_collection("LangChainCollection"):
    print("Collection exists")
else:
    client.drop_collection("LangChainCollection")

vectorstore = Milvus.from_documents(
    documents=pages,
    embedding=embeddings_model,
    connection_args={"uri": MILVUS_URL},
    index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
)
```

### 4. Hybrid Retrieval (BM25 + Vector Search)
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[vectorstore_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)
```

### 5. Define Prompt and Run Query
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub

prompt = ChatPromptTemplate.from_template(
    """
    <|system|>>
    You are a helpful AI Assistant that follows instructions extremely well.
    Use the following context to answer user question.
    
    CONTEXT: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
)

llm = HuggingFaceHub(
    repo_id="google/gemma-2b",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

chain = (
    {"context": ensemble_retriever, "query": lambda x: x} | prompt | llm
)
response = chain.invoke("Who is Frankenstein?")
print(response)
```

## Results
The model retrieves relevant context from the document and generates a well-informed response using the language model.

## License
This project is licensed under the MIT License.
