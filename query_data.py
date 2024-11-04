from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the Chroma database
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the database for similar content across all PDFs
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Print the embedding for the first result to understand what it looks like
    print("\nEmbeddings:")
    first_doc_embedding = embedding_function.embed_query(query_text)
    print(first_doc_embedding)  # Display embedding for the query

    # No filter for "monopoly" or other specific document sources, generalized
    # Create context from the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Display the context to understand what it includes
    print("\nContext for the query:")
    print(context_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Ollama model for text generation
    model = Ollama(model="mistral")

    # Generate the response using the context
    response_text = model.invoke(prompt)

    # Print the response and sources
    print("\nModel response received:")
    print(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    # Print the final response and sources
    print(formatted_response)

    return response_text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)
