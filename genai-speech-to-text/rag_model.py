import faiss
import numpy as np
import json
import os
import time
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === CONFIGURATION ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FOLDER = "faiss_index"
TOP_K = 1  # retrieve top 1 matching question
GEMINI_MODEL = "models/gemini-1.5-flash"

# === Load Gemini API ===
genai.configure(api_key="used_your_api_key_here")
gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)

# === Gemini Call ===
def call_gemini(prompt, retries=3, wait_seconds=15):
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Gemini rate limit hit. Retrying in {wait_seconds} sec...")
                time.sleep(wait_seconds)
            else:
                print(f"❌ Gemini error: {e}")
                break
    return "Sorry, I’m having trouble with our AI model right now. Please try again later or [Contact Support](https://themagnetismo.com/contact)."

# === RAG CLASS ===
class RAGModel:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(os.path.join(INDEX_FOLDER, "qa.index"))
        with open(os.path.join(INDEX_FOLDER, "qa_metadata.json"), "r", encoding="utf-8") as f:
            self.qa_data = json.load(f)

    def find_similar_answer(self, question):
        embedding = self.model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(embedding, TOP_K)
        closest_id = I[0][0]
        score = D[0][0]
        if closest_id < len(self.qa_data):
            return self.qa_data[closest_id], score
        return None, None

    def is_order_related(self, question):
        keywords = ['price', 'cost', 'buy', 'order', 'purchase']
        return any(word in question.lower() for word in keywords)

    def get_answer(self, user_question, use_gemini=True):
        matched_qa, score = self.find_similar_answer(user_question)

        if matched_qa:
            base_answer = matched_qa["answer"]

            if not use_gemini:
                final_answer = base_answer
            else:
                prompt = f"""
User asked: "{user_question}"

A similar known Q&A:
Q: {matched_qa['question']}
A: {base_answer}

Now generate a helpful and clear answer for the user's question based on this.
Avoid vague answers like "check website" or "depends" — instead, clearly say if product is available or link to contact/support/order page.
"""
                try:
                    final_answer = call_gemini(prompt).strip()

                    # Check for vague language
                    vague_phrases = ["might", "depends", "check", "could be", "we recommend", "typically", "usually"]
                    if any(phrase in final_answer.lower() for phrase in vague_phrases):
                        if self.is_order_related(user_question):
                            final_answer = (
                                "We're currently experiencing high demand, and availability may vary. "
                                "Please visit our [Order Page](https://themagtismo.com/pages/order) or "
                                "[Contact Support](https://themagtismo.com/contact) for up-to-date stock info and placing an order. 🛒"
                            )
                except Exception as e:
                    return f"⚠️ Gemini error: {str(e)}"
        else:
            final_answer = (
                "I'm not sure I have an answer for that. "
                "Could you ask something related to our product or services? 😊"
            )

        if self.is_order_related(user_question):
            final_answer += "\n\n🛒 [Order Now](https://themagtismo.com/pages/order) | 📞 [Contact Support](https://themagnetismo.com/contact)"

        return final_answer

# === Create RAG model instance ===
rag_model = RAGModel()

# === For terminal testing ===
if __name__ == "__main__":
    print("🔎 Retrieval-Augmented Q&A Chat")
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = rag_model.get_answer(q)
        print("\n🤖 Assistant:", answer)
