# app/utils/db_manager.py

from .vector_db import VectorDB
import os

class DBManager:
    _instance = None
    _vector_dbs = {}  # dict holding multiple database instances keyed by path
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_vector_db(self, vector_db_path, embedding_path):
        # Get the vector database instance
        # Use the path as the key to distinguish databases
        db_key = vector_db_path
        
        if db_key not in self._vector_dbs:
            self._vector_dbs[db_key] = VectorDB()
            if not os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
                print(f"First run, building database ({db_key})")
                self._vector_dbs[db_key].build_and_save(embedding_path, vector_db_path)
            else:
                print(f"Loading existing index ({db_key})")
                self._vector_dbs[db_key].load_index(vector_db_path)
                self._vector_dbs[db_key].to_gpu()
        
        return self._vector_dbs[db_key]