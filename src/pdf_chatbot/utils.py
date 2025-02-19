from pathlib import Path
from typing import List, Dict

import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document

from .preprocessing import Preprocessor, PreprocessorComposer


def get_pdf_files(dir_path: str, ext: str = "pdf") -> List[str]:
    """
    Recursively finds all PDF files in the given directory and its subfolders
    using glob and pathlib.
    """
    base_path = Path(dir_path)
    pdf_files = list(base_path.rglob(f"*.{ext}"))
    pdf_file_paths = [str(pdf) for pdf in pdf_files]
    return pdf_file_paths


def preprocess_pdf(
    file_path: str,
    preprocess: PreprocessorComposer | Preprocessor,
    start_page: int = None,
    end_page: int = None,
    add_meta: dict = {},
):

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()[start_page:end_page]

    for page in pages:
        page.metadata.update(add_meta)
        _ = preprocess(page, add_meta)

    return pages


def simple_format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def format_docs_with_xml_tags(docs: List[Document]) -> str:
    """Helperfunction: How to concat documents of the index db"""
    start_delimiter = "<source>\n"
    end_delimiter = "\n</source>"
    doc_txt_list = []
    for doc in docs:
        doc_txt = start_delimiter + doc.page_content + end_delimiter
        doc_txt_list.append(doc_txt)
    return "\n".join(doc_txt_list)


def get_bookmarks(filepath: str) -> Dict[int, str]:
    # WARNING! One page can have multiple bookmarks!
    bookmarks = {}
    with fitz.open(filepath) as doc:
        toc = doc.get_toc()  # [[lvl, title, page, …], …]
        for level, title, page in toc:
            bookmarks[page] = title
    return bookmarks
