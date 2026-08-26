from typing import List, Dict, Optional
import pandas as pd
from ir_framework.preprocessing.text_processor import TextProcessor
from ir_framework.data.dataset import IRDataset


class Vocabulary:
    """
    Manages vocabulary creation, indexing, and term-to-id mappings.
    Preserves exact term extraction and index mapping logic from ir_lib.
    """

    def __init__(self, vocabulary_list: List[str], pre_processing_mode: str = "none"):
        self.raw_vocabulary: List[str] = vocabulary_list
        self.pre_processing_mode: str = pre_processing_mode
        self.text_processor = TextProcessor(mode=pre_processing_mode)
        self._processed_vocabulary: Optional[List[str]] = None
        self._index_map: Optional[Dict[str, int]] = None

    @classmethod
    def build_vocabulary_from_tokens(cls, tokens: List[str]) -> List[str]:
        """Builds unique vocabulary from pre-sorted tokens."""
        vocabulary = []
        last_token = ""
        for token in tokens:
            if token != last_token:
                vocabulary.append(token)
                last_token = token
        return vocabulary

    @classmethod
    def from_corpus_text(cls, corpus_text: str, pre_processing_mode: str = "none") -> "Vocabulary":
        """Builds Vocabulary from full corpus text string."""
        processor = TextProcessor()
        tokens = processor.tokenize(corpus_text)
        raw_vocab = cls.build_vocabulary_from_tokens(tokens)
        return cls(raw_vocab, pre_processing_mode=pre_processing_mode)

    @classmethod
    def from_dataset(
        cls,
        dataset: IRDataset,
        text_column: Optional[str] = None,
        pre_processing_mode: str = "none"
    ) -> "Vocabulary":
        """Builds Vocabulary from an IRDataset."""
        col = text_column if text_column else dataset.text_column
        corpus_text = " ".join(dataset.df[col]).lower()
        return cls.from_corpus_text(corpus_text, pre_processing_mode=pre_processing_mode)

    @property
    def vocabulary(self) -> List[str]:
        """Returns the processed vocabulary according to pre_processing_mode."""
        if self._processed_vocabulary is None:
            if self.pre_processing_mode == "none":
                self._processed_vocabulary = self.raw_vocabulary
            elif self.pre_processing_mode == "no_stop_words":
                self._processed_vocabulary = self.text_processor.remove_stopwords(self.raw_vocabulary)
            elif self.pre_processing_mode == "stemming":
                self._processed_vocabulary = self.text_processor.stem(self.raw_vocabulary)
            elif self.pre_processing_mode == "lemmatization":
                self._processed_vocabulary = self.text_processor.lemmatize(self.raw_vocabulary)
            elif self.pre_processing_mode == "both":
                cleaned = self.text_processor.remove_stopwords(self.raw_vocabulary)
                self._processed_vocabulary = self.text_processor.stem(cleaned)
            else:
                self._processed_vocabulary = self.raw_vocabulary
        return self._processed_vocabulary

    @property
    def index_map(self) -> Dict[str, int]:
        """Returns a dict mapping term -> integer index."""
        if self._index_map is None:
            self._index_map = {term: idx for idx, term in enumerate(self.vocabulary)}
        return self._index_map

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __getitem__(self, term: str) -> int:
        return self.index_map[term]

    def __contains__(self, term: str) -> bool:
        return term in self.index_map

    def get(self, term: str, default: Optional[int] = None) -> Optional[int]:
        return self.index_map.get(term, default)
