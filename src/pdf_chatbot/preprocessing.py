import abc
import re

import fitz
import numpy as np
from typing import List
from pydantic import BaseModel


class Preprocessor(BaseModel, abc.ABC):

    def __call__(self, doc, *args, **kwargs):
        doc = self.preprocess_page(doc, *args, **kwargs)
        return doc
    
    @abc.abstractmethod
    def preprocess_page(self, doc, *args, **kwargs):
        ...       


class PreprocessorComposer(BaseModel):
    transforms: List[Preprocessor]
    
    def __call__(self, page, *args):
        for t in self.transforms:
            t(page, *args)
            
        return page


class RemoveHeader(Preprocessor):
    def preprocess_page(self, doc, *args, **kwargs):
        if "rm_header" not in doc.metadata:
            text = doc.page_content
            content = text.split("\n", 1)[1] #usually the first newline is the top line 
            doc.page_content = content
            doc.metadata.update(rm_header=True)
            
        return doc
    

class SimpleDehyphens(Preprocessor):
    def preprocess_page(self, doc, *args, **kwargs):
        text = doc.page_content
        content = text.replace(" -\n", "") # not optimal! 
        doc.page_content = content
        return doc
    
    
class RemoveStartingNumbers(Preprocessor):
    def preprocess_page(self, doc, *args, **kwargs):
        #TODO Do not remove sections numbers at the beginning
        text = doc.page_content
        # used gpt for the regex....looks a bit weird
        cleaned_string = re.sub(r'^(\d{3,}\.?\d*\s|\d{1,2}(?=\s)(?!\.\d+))', '', text)
        # Adjust to ensure non-valid single and double-digit numbers followed directly by letters are removed
        doc.page_content = re.sub(r'^(\d{1,2})(?=[^\d\s\.])', '', cleaned_string)
        
        return doc
    
    
class AddSectionNamesWithTOC(Preprocessor):
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
        #TODO handle pages where mutiples sections are mentioned
        sections = np.array(self.toc, dtype=object)
        section_pages = sections[:, 2]
        index = np.searchsorted(section_pages, page_number, side='right') - 1
        title = sections[index, 1]
        return title
    
    def preprocess_page(self, doc, *args, **kwargs):
        page_num = doc.metadata["page"]
        section_title = self._find_section_from_toc(page_num)
        splitted = section_title.split(" ", 1)

        if re.search(r"\d", splitted[0]):
            doc.metadata["section_title"] = splitted[1]
            doc.metadata["section_number"] = splitted[0]
        else:
            doc.metadata["section_title"] = section_title
            
        return doc