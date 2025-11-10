# RAG BYMA – Financial Retrieval-Augmented Generation System

This project implements a Retrieval-Augmented Generation (RAG) system designed to analyze financial reports of Argentine companies listed on BYMA (Bolsa y Mercados Argentinos).
It allows users to index quarterly PDF reports, retrieve relevant insights through natural language queries, and generate AI-powered responses grounded in the source documents.     

## Authors
* Zeballos Brenda
* Andrea Mujica
* Prata Emiliano
* Lisotti Joaquin

## Key Features

Automated PDF Indexing
Converts financial reports into semantically searchable text chunks optimized for embeddings.

Multilingual Semantic Search
Uses multilingual OpenAI embeddings and Cohere re-ranking to retrieve the most relevant sections, even for paraphrased or non-literal questions.

Context-Aware Response Generation
Produces fact-based, contextualized answers drawn directly from report content.

Automated Evaluation
Includes a benchmarking module to test the accuracy of generated answers against predefined question–answer sets.

Command-Line Interface (CLI)
Every function — indexing, querying, evaluation, and database reset — can be executed via simple CLI commands.


## Project Structure
```
rag_byma/
│
├── main.py                        # CLI entry point
├── create_parser.py               # Command parser for CLI
├── src/
│   ├── rag_pipeline.py            # Orchestrates the entire RAG flow
│   ├── impl/                      # Component implementations
│   │   ├── datastore.py           # Handles embeddings and vector storage
│   │   ├── indexer.py             # Extracts and chunks text from PDFs
│   │   ├── retriever.py           # Performs semantic search and reranking
│   │   ├── response_generator.py  # Generates final natural language answers
│   │   └── evaluator.py           # Evaluates accuracy and quality
│   ├── interface/                 # Base interfaces for extensibility
│
├── sample_data/
│   ├── source/                    # Example PDFs (e.g., YPF, Loma Negra)
│   └── eval/                      # Example Q&A sets for testing
│
└── README.md                      # Project documentation
```

## Installation
### 1. Clone the repository
```
git clone https://github.com/Lisotti/rag_byma.git
cd rag_byma
```

### 2. Set Up a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Configure Environment Variables
* LLM (OPENAI) you can modify/replace the LLM it in src/util/invoke_ai.py
```
export OPENAI_API_KEY='your_openai_api_key'
```
* COHERE
Set `CO_API_KEY="yoursupersecretapikey"` in .env file

## Usage
The CLI provides several commands to interact with the RAG pipeline. By default, they will use the source/eval paths specified in main.py, but there are flags to override them.
```
DEFAULT_SOURCE_PATH = "sample_data/source/"
DEFAULT_EVAL_PATH = "sample_data/eval/sample_questions.json"
```

### Run the Full Pipeline
reset datastore, index docs and evaluates
```
python main.py run
```
### Reset the Database
Clear the vector database
```
python main.py reset
```
### Add documents
```
python main.py add -p "sample_data/source/"
```
### Query the Database
```
python main.py query "What is the opening year of The Lagoon Breeze Hotel?"
```
### Evaluate the Model
```
python main.py evaluate -f "sample_data/eval/sample_questions.json"
```