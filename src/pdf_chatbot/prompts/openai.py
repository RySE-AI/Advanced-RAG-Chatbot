from langchain.prompts import ChatPromptTemplate

SIMPLE_RAG = """Answer the question based only on the following context:

{context}

Question: {question}
"""
SIMPLE_RAG_PROMPT = ChatPromptTemplate.from_template(SIMPLE_RAG)