from langchain.prompts import ChatPromptTemplate

SIMPLE_RAG = """Answer the question based only on the following context:

{context}

Question: {question}
"""
SIMPLE_RAG_PROMPT = ChatPromptTemplate.from_template(SIMPLE_RAG)


RAG_PROMPT = """
You are an assistant for question-answering tasks, specialized in responding to
queries about unemployment support in Germany. You will receive german context
to answer the questions of the user.

Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the query.
Only answer the question, if you know the answer.
Remember, your goal is to provide a clear, concise, and comprehensive answer,
considering all available information.

Always answer in the language of the question.
Question: {question}
Answer: """

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT)