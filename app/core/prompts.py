"""Prompt templates used by the RAG pipeline."""

rag_question_prompt: str = """
Based on the following context, answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
"""
