# RAG LLM PDF Parser Using Ollama Model

## Description

The **RAG LLM PDF Parser Using Ollama Model** is a project designed to enable users to query information from PDFs stored in a local database, powered by an Ollama language model and Chroma vector store. This tool provides relevant responses to questions based on the content of the PDFs, leveraging dense embeddings for similarity search. The solution is especially useful for retrieving targeted information from a large collection of PDF documents, enabling users to query and validate responses effectively.

The workflow consists of two main stages:
1. **Data Processing and Storage**: PDFs are loaded, split into smaller chunks, embedded, and stored in a Chroma vector store.
2. **Query and Response Generation**: Queries are executed against the Chroma vector store, retrieving relevant content which is then fed into the Ollama model to generate responses.

### Code Structure and Execution Flow
- **`database_setup.py`**: This file initializes the database, loads PDF documents, splits them into smaller text chunks, and adds these chunks to the Chroma vector store. It begins by checking if the database needs a reset and then proceeds with data processing and storage.
  - **Starting Point**: This script should be run first to prepare the data for querying.
  - **Command to Run**: `python database_setup.py --reset` (optional `--reset` to clear and reload the database).
  
- **`query_data.py`**: This file contains the `query_rag` function, which is used to handle queries. It retrieves relevant data chunks from the Chroma vector store, formats them into a prompt, and generates a response using the Ollama model.
  - **Command to Run**: `python query_data.py "Your query here"`.

- **`test_script.py`**: A testing file that runs predefined queries to validate the responses from the Ollama model against expected answers. This script uses evaluation prompts to verify the accuracy of the responses.
  - **Command to Run**: `python test_script.py`.
  
- **`get_embedding_function.py`**: A helper script that defines the embedding function used for generating vector embeddings, essential for querying relevant data chunks.

Each file works in coordination, with `database_setup.py` setting up the data storage, `query_data.py` handling the question-answering process, and `test_script.py` allowing for validation of the model’s output.

### Sequence of Execution
1. **Run `database_setup.py`** to initialize the Chroma database, load PDFs, split documents, and add them to the vector store.
2. **Use `query_data.py`** to query specific questions and retrieve responses based on the stored PDF data.
3. **Run `test_script.py`** (optional) to test specific queries and validate the accuracy of the responses.

## Requirements

- **Python 3.8+**
- **Git**: To clone the repository and manage version control.

## Dependencies

The following Python libraries are required:
- `langchain_community`: Provides tools for document loading, LLM management, and vector storage.
- `langchain`: Core library for handling prompts, LLM models, and embedding functions.
- `Chroma`: Vector database to store document embeddings for similarity search.
- `Ollama`: Language model library for generating responses.
- `argparse`: Built-in Python library for handling command-line arguments.
- `shutil`: Built-in library for file operations (used for clearing database).

### Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/AdityaShah346/Ollama_RAG_Reader.git
   cd Ollama_RAG_Reader
   ```
   
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Create necessary directories:
   ```bash
   mkdir chroma data
   ```
   
4. Place PDF files you want to query in the `data` directory.

## Usage

### Step 1: Load and Process Data
Run `database_setup.py` to load PDFs, split them, and store embeddings in the Chroma vector database:
```bash
python database_setup.py --reset
```

### Step 2: Query Data
Use `query_data.py` to ask questions based on the PDF contents:
```bash
python query_data.py "Your query text here"
```

### Step 3: Run Tests (Optional)
Run `test_script.py` to validate the responses for specific queries:
```bash
python test_script.py
```

## Contributor

- **Aditya Shah**: Developed the entire project. [GitHub Profile](https://github.com/AdityaShah346)

## Additional Information

This project is a standalone PDF parser using RAG and the Ollama model to streamline information retrieval from documents. It’s designed for ease of use and customization. Future improvements may include expanding supported document types or optimizing embedding functions for specific domains.
```
