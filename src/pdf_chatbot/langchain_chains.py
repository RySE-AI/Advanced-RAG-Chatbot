from langchain_core.runnables import chain
from langdetect import detect_langs
from operator import itemgetter

from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
)
from deep_translator import GoogleTranslator
from pdf_chatbot.retriever import MultiQueryRetrieverWithQueries


@chain
def detect_german_language(text: str, threshold: float = 0.7):
    detected_lngs = detect_langs(text)
    top_lng = detected_lngs[0]
    threshold_cnd = top_lng.lang == "de" and top_lng.prob < threshold

    if top_lng.lang != "de" or threshold_cnd:
        return True


def translate_to_german(text: str):
    translated = GoogleTranslator(source="auto", target="de").translate(text)
    return translated


def create_naive_retriever_chain(retriever, format_docs, prompt, llm, parser):
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return qa_chain


def create_multi_query_retriever(retriever, format_docs, prompt, llm, parser):
    mq_retriever = MultiQueryRetrieverWithQueries.from_llm(
        retriever=retriever, llm=llm, include_original=True, return_queries=True
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"]["documents"]))
        )
        | prompt
        | llm
        | parser
    )

    rag_chain_with_source = RunnableParallel(
        {"context": mq_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def create_self_query_retriever(retriever, format_docs, prompt, llm, parser):
    pass


def translation_branch_chain():
    @chain
    def _route(info):
        if info["translate"]:
            return translate_to_german(info["question"])
        else:
            return info["question"]

    translate_chain = {
        "translate": detect_german_language,
        "question": RunnablePassthrough(),
    } | _route

    return translate_chain


def create_multi_query_retriever_with_translator(
    retriever, format_docs, prompt, llm, parser
):
    mq_retriever = MultiQueryRetrieverWithQueries.from_llm(
        retriever=retriever, llm=llm, include_original=True, return_queries=True
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"]["documents"]))
        )
        | prompt
        | llm
        | parser
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": translation_branch_chain() | mq_retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source
