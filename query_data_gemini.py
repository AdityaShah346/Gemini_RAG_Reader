import argparse
import PyPDF2
import google.generativeai as genai
from PyPDF2.errors import PdfReadError

# Set your Google Gemini API key here
API_KEY = "AIzaSyDnNEYbVL8UfBw5MkaYPTxeGfwVY6oxewQ"

# Configure the generative AI model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

MAX_CHUNK_SIZE = 50000  # Set a limit on chunk size for sending to API

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path, password=None):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Decrypt the PDF if needed
        if reader.is_encrypted:
            try:
                reader.decrypt(password or "")
                print("PDF Decrypted successfully!")
            except PdfReadError as e:
                print(f"Failed to decrypt PDF: {e}")
                return ""

        # Extract the text from all the pages
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Function to split text into manageable chunks
def split_text_into_chunks(text, max_size):
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for paragraph in text.split('\n'):
        paragraph_size = len(paragraph.encode('utf-8'))

        if current_chunk_size + paragraph_size > max_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_chunk_size = 0

        current_chunk.append(paragraph)
        current_chunk_size += paragraph_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# Function to query the Gemini model using `google-generativeai`
def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error while querying Gemini: {e}")
        return None

# Function to process the PDF and send queries
def query_pdf(pdf_path, query_text, password=None):
    # Extract the text from the PDF
    context = extract_text_from_pdf(pdf_path, password)

    # Split the text into smaller chunks to avoid exceeding the API size limit
    chunks = split_text_into_chunks(context, MAX_CHUNK_SIZE)

    # Iterate over chunks and send them to Gemini API
    for idx, chunk in enumerate(chunks):
        prompt = f"""
        Based on the following context, answer the question: {query_text}
        Context: {chunk}
        """
        print(f"Querying chunk {idx + 1} of {len(chunks)}...")
        response_text = query_gemini(prompt)
        print(f"Response for chunk {idx + 1}:", response_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--password", type=str, default=None, help="Password for encrypted PDF (if any).")
    args = parser.parse_args()

    query_pdf(args.pdf_path, args.query_text, password=args.password)


# api_key = "AIzaSyDPTAqbD1yWLLsiE7S2NaaPj0Ek4Z61E-I"

