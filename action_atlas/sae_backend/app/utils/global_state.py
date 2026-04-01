# app/utils/global_state.py
from datetime import datetime

class GlobalState:
    _instance = None
    
    def __init__(self):
        self._current_query = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def save_current_query(self, query_text, query_vector):
        """保存当前查询"""
        self._current_query = {
            'query_text': query_text,
            'query_vector': query_vector,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def get_current_query(self):
        """获取当前查询"""
        return self._current_query