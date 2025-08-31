# 1_generate_embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

# 📂 Input Q&A file (plain text)
QA_FILE = "product_qa.txt"
INDEX_FOLDER = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load sentence transformer
model = SentenceTransformer(EMBEDDING_MODEL)

def parse_qa_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    qa_pairs = []
    question, answer = "", ""

    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question and answer:
                qa_pairs.append({"question": question, "answer": answer})
                question, answer = "", ""

    return qa_pairs

def create_faiss_index(qa_pairs, save_path):
    questions = [item["question"] for item in qa_pairs]
    embeddings = model.encode(questions, convert_to_numpy=True)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    faiss.write_index(index, os.path.join(save_path, "qa.index"))

    # Save metadata
    with open(os.path.join(save_path, "qa_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"[✅] FAISS index and metadata saved to: {save_path}")

if __name__ == "__main__":
    print("[📄] Reading Q&A file...")
    qa_pairs = parse_qa_file(QA_FILE)
    print(f"[📌] Parsed {len(qa_pairs)} Q&A pairs.")
    create_faiss_index(qa_pairs, INDEX_FOLDER)
