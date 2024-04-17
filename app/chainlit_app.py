import chainlit as cl
import chromadb

from chainlit.input_widget import Select, Slider
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema.runnable import Runnable, RunnableConfig

from pdf_chatbot.langchain_chains import create_multi_query_retriever_with_translator
from pdf_chatbot.utils import simple_format_docs
from pdf_chatbot.prompts.openai import SIMPLE_RAG_PROMPT


CHROMA_CLIENT = chromadb.HttpClient(host="localhost", port=8000)

# Settings Widgets
input_widgets = [
    Select(
        id="model",
        label="OpenAI - Model",
        values=["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-turbo-2024-04-09"],
        initial_index=0,
    ),
    Select(
        id="collection_name",
        label="Name of the embedding collection",
        values=["openai_embedded", "multi-qa-mpnet-base-base_embedded"],
        initial_index=0,
    ),
    Select(
        id="filter",
        label="Filtering the documents",
        values=["all", "Arbeitslosengeld", "Bürgergeld"],
        initial_index=0,
    ),
    Slider(
        id="temperature",
        label="OpenAI - Temperature",
        initial=0.0,
        min=0,
        max=2,
        step=0.1,
    ),
    Slider(
        id="seed",
        label="Seed for the openai model",
        initial=30,
        min=0,
        max=150,
        step=1,
        description="Set a seed for the Openai model",
    ),
]


@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(input_widgets).send()
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    # Setup embedding model depending on collection
    if settings["collection_name"] == "openai_embedded":
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    elif settings["collection_name"] == "multi-qa-mpnet-base-base_embedded":
        embedding_model = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    else:
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Setup Collection
    database = Chroma(
        client=CHROMA_CLIENT,
        collection_name=settings["collection_name"],
        embedding_function=embedding_model,
    )

    # Setup OpenAI Model
    model = ChatOpenAI(
        model_name=settings["model"],
        temperature=settings["temperature"],
        streaming=True,
        model_kwargs={"seed": int(settings["seed"])},
    )

    runnable = create_multi_query_retriever_with_translator(
        retriever=database.as_retriever(),
        format_docs=simple_format_docs,
        prompt=SIMPLE_RAG_PROMPT,
        llm=model,
        parser=StrOutputParser(),
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    # res = await runnable.ainvoke(message.content)
    # msg = cl.Message(content=res["answer"])
    # await msg.send()

    # elements = [
    #     cl.Pdf(
    #         name="pdf1",
    #         display="side",
    #         path="pdf_files/merkblatt-buergergeld_ba043375.pdf",
    #     )
    # ]
    # # Reminder: The name of the pdf must be in the content of the message
    # cl.Message(content="Look at this local pdf1!", elements=elements).send()

    msg = cl.Message(content="")
    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()]),
        ):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
            if "context" in chunk:
                context = chunk["context"]
            if "question" in chunk:
                original_question = chunk["question"]

    # msg.content = context["documents"]
    # await msg.send()
