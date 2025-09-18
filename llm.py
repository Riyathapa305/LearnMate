from typing import List
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from vectorstore import VectorStore
from memory import ChatMemory
from pdf import load_and_split_pdf

class LLM:
    def __init__(self, vectorstore, chat_memory, model_name: str = "gemini-1.5-flash"):
        
        self.model = GoogleGenerativeAI(model=model_name)  # temperature can be set here if needed
        self.vectorstore = vectorstore
        self.chat_memory = chat_memory
        self.pdf_active = False


    def upload_pdf(self, pdf_path: str):
        try:
            chunks = load_and_split_pdf(pdf_path)
            if not chunks:
                return "PDF has no content."
            self.vectorstore.add_document(chunks)  # pass strings directly
            self.pdf_active = True
            return "PDF uploaded successfully"
        except Exception as e:
            return f"Failed to upload PDF: {e}"


    def clear_pdf(self):
        # Resets PDF documents but preserves chat memory
        self.vectorstore = VectorStore()  
        self.pdf_active = False
        self.chat_memory.clear()

        return "PDF cleared. Back to general chat."

    def generate_response(self, prompt: str):
        try:
            response = self.model.invoke(messages=[{"role": "user", "content": prompt}])
            
            return response.content  
        except Exception as e:
            return f"LLM Error: {e}"
    def generate_response(self, prompt: str):
        response = self.model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def answer_query(self, user_input: str, top_k: int = 3):
        short_context = self.chat_memory.get_context()
        retrieval_context = ""
        retrieved_chunks = []

        if self.pdf_active:
            retrieved_chunks = self.vectorstore.search(user_input, top_k=top_k)
            retrieval_context = "\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful assistant.
Conversation so far:
{short_context}

Relevant knowledge:
{retrieval_context if retrieval_context else "N/A"}

User question:
{user_input}

Provide a clear and useful answer.
"""

        assistant_reply = self.generate_response(prompt)

        # short-term memory
        self.chat_memory.add_message(user_input, assistant_reply)

        return assistant_reply, retrieved_chunks
