import abc
import re

import fitz
import numpy as np
from typing import List
from pydantic import BaseModel

from langchain_core.documents import Document


class Preprocessor(BaseModel, abc.ABC):
    """Abstract base class to create a Langchain Document preprocessor.
    Inherit from the class and change the preprocess method.

    Example:
    class AddABC123ToMetadata(Preprocessor):
        def preprocess_page(self, doc, *args, **kwargs):
            doc.metadata.update({"ABC": 123})
            return doc

    for doc in documents: # LangChain Documents
        doc = AddABC123ToMetadata(doc)
    """

    def __call__(self, doc: Document, *args, **kwargs) -> Document:
        doc = self.preprocess_page(doc, *args, **kwargs)
        return doc

    @abc.abstractmethod
    def preprocess_page(self, doc: Document, *args, **kwargs) -> Document: ...


class PreprocessorComposer(BaseModel):
    """Composes several preprocessors together.
    
    Args:
        preprocessors (List[Preprocessor]): A list of preprocessors which will be
        applied sequentially to a document.
        
    Example:
        compose = PreprocessorComposer(    
                preprocessors=[
                            RemoveHeader(),
                            SimpleDehyphens(),
                            RemoveStartingNumbers(),
                            AddSectionNamesWithTOC(file_path=file_path)])
                
        for doc in documents: # LangChain Documents
            doc = compose(doc)
    """

    preprocessors: List[Preprocessor]

    def __call__(self, doc: Document, *args):
        for t in self.preprocessors:
            t(doc, *args)

        return doc


class RemoveHeader(Preprocessor):
    """Removes the first line of the page content of a document"""
    def preprocess_page(self, doc: Document, *args, **kwargs) -> Document:
        if "rm_header" not in doc.metadata:
            text = doc.page_content
            content = text.split("\n", 1)[
                1
            ]  # usually the first newline is the top line
            doc.page_content = content
            doc.metadata.update(rm_header=True)

        return doc


class SimpleDehyphens(Preprocessor):
    """'Silbentrennung': Simply removes Hyphens between words which are
    separated by a newline"""
    def preprocess_page(self, doc: Document, *args, **kwargs) -> Document:
        text = doc.page_content
        content = text.replace(" -\n", "")  # not optimal!
        doc.page_content = content
        return doc


class RemoveStartingNumbers(Preprocessor):
    """Special for the PDF Documents of the project. This removes 'weird'
    numbers at the beginning of Document."""
    def preprocess_page(self, doc: Document, *args, **kwargs) -> Document:
        # TODO Do not remove sections numbers at the beginning
        text = doc.page_content
        # used gpt for the regex....looks a bit weird
        cleaned_string = re.sub(r"^(\d{3,}\.?\d*\s|\d{1,2}(?=\s)(?!\.\d+))", "", text)
        # Adjust to ensure non-valid single and double-digit numbers followed directly by letters are removed
        doc.page_content = re.sub(r"^(\d{1,2})(?=[^\d\s\.])", "", cleaned_string)

        return doc


class AddSectionNamesWithTOC(Preprocessor):
    """Add section titles to document's metadata extracted from the table of 
    content of a pdf file.

    Args:
        file_path (str): path to the pdf file where the toc will be extracted
        toc (List[List]): 
            Pass a table of content in the form of fitz .get_toc() method, only
            necessary if the pdf file doesn't have a toc!
    """
    file_path: str
    toc: List[List] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.toc = self._load_file_content()

    def _load_file_content(self):
        if self.file_path:
            try:
                with fitz.open(self.file_path) as doc:
                    return doc.get_toc()
            except Exception as e:
                raise ValueError(f"Failed to read from {self.file_path}: {str(e)}")

    def _find_section_from_toc(self, page_number):
        # TODO handle pages where mutiples sections are mentioned
        sections = np.array(self.toc, dtype=object)
        section_pages = sections[:, 2]
        index = np.searchsorted(section_pages, page_number, side="right") - 1
        title = sections[index, 1]
        return title

    def preprocess_page(self, doc: Document, *args, **kwargs) -> Document:
        page_num = doc.metadata["page"]
        section_title = self._find_section_from_toc(page_num)
        splitted = section_title.split(" ", 1)

        if re.search(r"\d", splitted[0]):
            doc.metadata["section_title"] = splitted[1]
            doc.metadata["section_number"] = splitted[0]
        else:
            doc.metadata["section_title"] = section_title

        return doc
