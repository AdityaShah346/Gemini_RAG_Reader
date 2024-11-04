import argparse
import os
from langchain_community.document_loaders import PyPDFLoader  # Updated for single PDF loading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma  # Updated Chroma import
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

# Chroma Database Directory (this will hold the vector store)
CHROMA_PATH = "chroma_db"

# Prompt Template for Querying the Model
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    """Get embedding function using Ollama embeddings."""
    return OllamaEmbeddings(model="nomic-embed-text")


def load_pdf(pdf_path: str):
    """Load and split PDF document into chunks."""
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def add_to_chroma(chunks: list[Document]):
    """Add document chunks to Chroma vector store."""
    # Initialize Chroma vector store
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Assign IDs to each chunk
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add documents to the vector store
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} new document chunks to the database.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new document chunks to add.")


def calculate_chunk_ids(chunks):
    """Calculate chunk IDs for documents."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def query_rag(query_text: str):
    """Query the Chroma vector store and use Ollama model for answering."""
    # Initialize Chroma vector store
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for relevant document chunks in the vector store
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("No relevant context found. Try a different query.")
        return

    # Create context from the top results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Ollama model for text generation
    model = Ollama(model="mistral")

    # Generate and print the response
    response_text = model.invoke(prompt)
    print("Model response received:", response_text)

    sources = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="PDF Query Tool using Chroma and Ollama")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file or directory")
    parser.add_argument("query_text", type=str, help="The query text to ask the PDF")
    parser.add_argument("--reset", action="store_true", help="Clear the Chroma vector store")
    args = parser.parse_args()

    if args.reset:
        print("✨ Resetting the Chroma vector store")
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)

    # Step 1: Load PDF and split into chunks
    print(f"📄 Loading PDF from: {args.pdf_path}")
    chunks = load_pdf(args.pdf_path)

    # Step 2: Add chunks to Chroma database
    add_to_chroma(chunks)

    # Step 3: Query the Chroma database
    print(f"❓ Asking: {args.query_text}")
    query_rag(args.query_text)
