import os
import re
import time
from typing import List

import faiss
import gradio as gr
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer  # For embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # For progress bars

# LLM
from transformers import pipeline

# Data Directory
DATA_DIR = "pdfs"  # Relative path (assuming 'pdfs' directory is in the repo)

# Model Names
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # A good balance of speed and quality
LLM_MODEL_NAME = "google/flan-t5-base"  # Or a smaller one for faster inference
MAX_CONTEXT_LENGTH = 512  # Maximum context length for the LLM

# Sample Queries
SAMPLE_QUERIES = [
    "What was Google's total revenue in 2023?",
    "How did Google's research and development expenses change between 2023 and 2024?",
    "What is the capital of France?",
]

def load_pdfs(data_dir: str) -> List[str]:
    """Loads and extracts text from PDF files in a directory."""
    pdf_texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    reader = PdfReader(f)
                    text = "".join([p.extract_text() for p in reader.pages])
                    pdf_texts.append(text)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return pdf_texts


def clean_text(text):
    """Clean the financial text by removing extra spaces, newlines, and special characters."""
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('$', '')  # Remove currency symbols
    text = text.encode('ascii', 'ignore').decode()  # Remove non-ASCII characters
    return text.strip()


def chunk_text(text, chunk_size=512, overlap=50):
    """Splits text into overlapping chunks."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Split into sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= chunk_size:  # Check for number of words in a chunk
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())
    # Overlap chunks
    overlapped_chunks = []
    for i in range(len(chunks)):
        overlapped_chunks.append(chunks[i])

    return overlapped_chunks


def bm25_retrieval(query, tfidf_matrix, vectorizer, top_k=5):
    """Retrieves relevant chunks using BM25."""
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    relevant_indices = similarity_scores.argsort()[-top_k:][::-1]  # Get top_k indices
    return relevant_indices, similarity_scores[relevant_indices]


def hybrid_retrieval(query, tfidf_matrix, tfidf_vectorizer, faiss_index, embedding_model, chunks, top_k=5):
    """Combines BM25 and embedding retrieval, then re-ranks."""

    # BM25 Retrieval
    bm25_indices, bm25_scores = bm25_retrieval(query, tfidf_matrix, tfidf_vectorizer, top_k=5)

    # Embedding Retrieval
    query_embedding = embedding_model.encode(query)
    distances, embedding_indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    embedding_indices = embedding_indices.flatten()
    embedding_scores = 1 - distances.flatten() / 2  # Convert distance to similarity-like score

    # Combine Results (You can experiment with different weighting strategies)
    combined_indices = list(bm25_indices) + list(embedding_indices)
    combined_scores = list(bm25_scores) + list(embedding_scores)

    # Remove duplicates
    unique_indices = []
    unique_scores = []
    seen = set()
    for i, index in enumerate(combined_indices):
        if index not in seen:
            unique_indices.append(index)
            unique_scores.append(combined_scores[i])
            seen.add(index)

    # Re-ranking (Simple: just sort by combined score)
    ranked_results = sorted(zip(unique_indices, unique_scores), key=lambda x: x[1], reverse=True)[:top_k]
    ranked_indices = [index for index, score in ranked_results]
    ranked_scores = [score for index, score in ranked_results]  # get the scores

    return ranked_indices, ranked_scores


def generate_answer(query, context, qa_pipeline):
    """Generates an answer from the context using the LLM. Truncates context if too long."""
    prompt = f"Question: {query} Context: {context} Answer:"

    # Truncate the prompt if it exceeds the maximum context length
    if len(prompt.split()) > MAX_CONTEXT_LENGTH:
        prompt_words = prompt.split()
        prompt = " ".join(prompt_words[:MAX_CONTEXT_LENGTH])  # Keep only the first MAX_CONTEXT_LENGTH words

    try:
        result = qa_pipeline(prompt, max_length=512, min_length=50, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while generating the answer."


def input_guardrail(query):
    """Checks if the query is safe and relevant."""
    # Enhanced: Check for financial keywords and relevance
    query = query.lower()
    if not any(keyword in query for keyword in ["revenue", "profit", "loss", "cash flow", "balance sheet", "financial", "income", "expense"]):
        if any(keyword in query for keyword in ["capital of France", "irrelevant", "harmful"]):
            return "I cannot answer questions of that nature. Please ask a relevant financial question.", 0.0, 0, "Irrelevant/Harmful Query", EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
        else:
            return "Please ask a specific question related to financial information.", 0.0, 0, "Non-Financial Query", EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
    return None, None, None, "Safe", None, None  # Query is safe


def rag_pipeline(query, tfidf_matrix, tfidf_vectorizer, faiss_index, embedding_model, chunks, qa_pipeline):
    """Executes the RAG pipeline."""
    start_time = time.time()

    # 1. Retrieval
    relevant_indices, retrieval_scores = hybrid_retrieval(query, tfidf_matrix, tfidf_vectorizer, faiss_index, embedding_model, chunks, top_k=5)  # Adjust top_k as needed
    context = "\n".join([chunks[i] for i in relevant_indices])

    # 2. Generation
    answer = generate_answer(query, context, qa_pipeline)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate Confidence Score (Example)
    if retrieval_scores:
        confidence_score = np.mean(retrieval_scores)  # Average retrieval score
    else:
        confidence_score = 0.0  # If no context retrieved, low confidence

    return answer, elapsed_time, confidence_score, len(chunks)  # Return number of chunks


def answer_question(query, tfidf_matrix, tfidf_vectorizer, faiss_index, embedding_model, chunks, qa_pipeline):
    """Main function to run the RAG pipeline and return the answer and confidence."""

    # Guardrail check
    guardrail_message, guardrail_confidence, num_chunks, guardrail_reason, emb_model, llm_model = input_guardrail(query)
    if guardrail_message:
        output_text = (
            f"<div style='color:red;'><b>Guardrail Triggered:</b> {guardrail_reason}</div>"
            f"<div style='color:red;'><b>Message:</b> {guardrail_message}</div>"
            f"<div><b>Embedding Model:</b> {emb_model}</div>"
            f"<div><b>LLM Model:</b> {llm_model}</div>"
        )
        return (output_text, guardrail_confidence, num_chunks, guardrail_reason, emb_model, llm_model, "N/A")  # Explicitly create a tuple

    answer, elapsed_time, confidence_score, num_chunks = rag_pipeline(query, tfidf_matrix, tfidf_vectorizer, faiss_index, embedding_model, chunks, qa_pipeline)

    output_text = (
        f"<div><b>Answer:</b> {answer}</div>"
        f"<div><b>Elapsed Time:</b> {elapsed_time:.4f} seconds</div>"
        f"<div><b>Confidence:</b> {confidence_score:.2f}</div>"
        f"<div><b>Chunks Scanned:</b> {num_chunks}</div>"
        f"<div><b>Guardrail Status:</b> Safe Query</div>"
        f"<div><b>Embedding Model:</b> {EMBEDDING_MODEL_NAME}</div>"
        f"<div><b>LLM Model:</b> {LLM_MODEL_NAME}</div>"
    )
    return (output_text, elapsed_time, confidence_score, num_chunks, "Safe Query", EMBEDDING_MODEL_NAME, LLM_MODEL_NAME)


if __name__ == "__main__":
    # 1. Load PDFs
    pdf_texts = load_pdfs(DATA_DIR)
    print(f"Loaded {len(pdf_texts)} PDFs.")

    # 2. Clean Text
    cleaned_texts = [clean_text(text) for text in pdf_texts]
    print("Text cleaning complete.")

    # 3. Chunk Text
    chunks = []
    for text in cleaned_texts:
        chunks.extend(chunk_text(text))
    print(f"Created {len(chunks)} chunks.")

    # 4. Create Embeddings and FAISS Index
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for cosine similarity
    faiss_index.add(chunk_embeddings)
    print("Embeddings created and FAISS index built.")

    # 5. BM25 Implementation
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)

    # 6. Initialize LLM pipeline
    qa_pipeline = pipeline("text2text-generation", model=LLM_MODEL_NAME)

    # 7. Gradio UI

    def ui_fn(query):
        return answer_question(query, tfidf_matrix, tfidf_vectorizer, faiss_index, embedding_model, chunks, qa_pipeline)

    with gr.Blocks(title="Financial RAG Assistant") as iface:
        gr.Markdown("# Financial RAG Assistant")
        gr.Markdown("Ask questions about company financial statements.")

        with gr.Row():
            query_input = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Your Question")

        with gr.Row():
            sample_query_buttons = [gr.Button(value=query) for query in SAMPLE_QUERIES]

        with gr.Row():
            submit_button = gr.Button("Submit")

        with gr.Accordion("Detailed Results", open=False):
            answer_output = gr.HTML(label="Answer")
            elapsed_time_output = gr.Number(label="Elapsed Time")
            confidence_output = gr.Number(label="Confidence")
            chunks_scanned_output = gr.Number(label="Chunks Scanned")
            guardrail_status_output = gr.Textbox(label="Guardrail Status")
            embedding_model_output = gr.Textbox(label="Embedding Model")
            llm_model_output = gr.Textbox(label="LLM Model")


        submit_button.click(
            fn=ui_fn,
            inputs=query_input,
            outputs=[answer_output, elapsed_time_output, confidence_output, chunks_scanned_output, guardrail_status_output, embedding_model_output, llm_model_output],
        )

        # The correct way to handle sample query buttons:
        def set_query(query):
            return query
        for button in sample_query_buttons:
            button.click(
                fn=set_query,
                inputs=button,  # Pass the button itself as input
                outputs=query_input,
            )

    iface.launch(debug=True)