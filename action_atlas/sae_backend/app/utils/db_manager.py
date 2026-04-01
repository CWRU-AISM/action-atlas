# app/utils/db_manager.py

from .vector_db import VectorDB
import os

class DBManager:
    _instance = None
    _vector_dbs = {}  # 改为字典，用于存储多个数据库实例
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_vector_db(self, vector_db_path, embedding_path):
        """获取向量数据库实例"""
        # 使用路径作为key来区分不同的数据库
        db_key = vector_db_path
        
        if db_key not in self._vector_dbs:
            self._vector_dbs[db_key] = VectorDB()
            if not os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
                print(f"首次运行，构建数据库... ({db_key})")
                self._vector_dbs[db_key].build_and_save(embedding_path, vector_db_path)
            else:
                print(f"加载现有索引... ({db_key})")
                self._vector_dbs[db_key].load_index(vector_db_path)
                self._vector_dbs[db_key].to_gpu()
        
        return self._vector_dbs[db_key]