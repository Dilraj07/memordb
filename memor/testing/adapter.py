from abc import ABC, abstractmethod
from typing import List, Dict

class MemoryBackendAdapter(ABC):
    """
    Abstract interface that all memory backends must implement to be tested 
    by the temporal memory benchmark suite.
    """

    @abstractmethod
    def add_fact(self, user_id: str, text: str) -> None:
        """
        Add a factual statement to the memory backend synchronously.
        The backend should extract and store the facts within this call.
        """
        pass

    @abstractmethod
    def get_all_active_facts(self, user_id: str) -> List[Dict[str, str]]:
        """
        Return a list of dictionaries for all currently active facts.
        Each dictionary must contain at least 'predicate' and 'object' keys.
        """
        pass
        
    @abstractmethod
    def query(self, user_id: str, query_text: str, k: int = 3) -> List[str]:
        """
        Query the memory backend and return a list of contextual strings.
        These strings should contain the facts relevant to the query_text.
        """
        pass

    @abstractmethod
    def clear(self, user_id: str) -> None:
        """
        Clear all facts for the given user_id. 
        Used to reset state between tests.
        """
        pass
