# RSS-MCP: Knowledge Graph RAG System

RSS-MCP (Retrieval-Augmented Generation System with Modular Component Pipeline) is a full-stack application that builds and queries a knowledge graph from your documents. It uses a sophisticated hybrid retrieval mechanism, combining semantic vector search with structured graph queries to provide accurate, context-aware answers.

The entire application was developed following strict Test-Driven Development (TDD) principles, with a modular backend and an interactive Streamlit user interface.

## ✨ Features

- **Document Ingestion**: Upload Markdown documents through a web UI. The system automatically processes the text, extracts entities and relationships, and stores them in a Neo4j knowledge graph.
- **Hybrid RAG Chat**: Ask questions in natural language. The system uses a combination of vector similarity search and graph traversal to find the most relevant information and generate a coherent answer.
- **Answer Context Visualization**: For each answer, the UI displays the specific subgraph (nodes and relationships) that was used as the primary context, providing explainability for the AI's response.
- **Full Graph Visualization**: Interactively explore the entire knowledge graph stored in the database.
- **RAG Evaluation Dashboard**: Quantitatively measure the performance of the RAG pipeline using metrics like faithfulness and answer relevancy, powered by the `ragas` library.

## 🛠️ Tech Stack

- **Backend**: Python, LangChain, TDD (pytest)
- **LLMs**: Designed for modularity (Ollama, OpenAI). Currently configured for Ollama.
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Database**: Neo4j (for knowledge graph and vector index)
- **Frontend**: Streamlit
- **Environment**: `uv` for package management, Docker for database setup

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose
- [Python 3.9+](https://www.python.org/downloads/)
- `uv` (Python package manager): `pip install uv`

### 1. Set up Environment Variables

Copy the example environment file and fill in your details if necessary. The defaults are set up for local development.

```bash
cp .env.example .env
```

### 2. Start the Neo4j Database

Run the Neo4j database instance using Docker Compose. This will start a container and persist data in a Docker volume named `neo4j_data`.

```bash
docker-compose up -d
```
The Neo4j Browser will be available at `http://localhost:7474`. You can log in with the credentials from your `.env` file (default: `neo4j`/`password`).

### 3. Install Dependencies

Create a virtual environment and install all required packages using `uv`.

```bash
# Create the virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip sync requirements.txt

# Install the project in editable mode
uv pip install -e .
```

### 4. Run the Streamlit Application

Launch the web application.

```bash
streamlit run app.py
```

The application will now be running at `http://localhost:8501`.

##  Usage

1.  **Navigate to the App**: Open your web browser to `http://localhost:8501`.
2.  **Ingest a Document**: Go to the "Ingest Document" page from the sidebar and upload a Markdown file (you can use `sample_document.md` from this repository).
3.  **Chat with the Graph**: Go to the "Chat with Graph" page and ask questions about the document you just ingested (e.g., "Who is Jules?").
4.  **Visualize**: Explore the full graph on the "Visualize Graph" page or see the answer context in the chat.
5.  **Evaluate**: Run the RAGAS evaluation on the "Evaluation Dashboard" page.
