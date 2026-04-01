import numpy as np
import faiss
import glob
import time
from tqdm import tqdm
import gc
import pickle
import os
from datetime import datetime
import json
import concurrent

class VectorDB:
    def __init__(self):
        self.index = None
        self.gpu_index = None
        self.descriptions = []
        self.indices = []
        self.file_mappings = [] 
        self.dimension = 3072
        self.res = None
        self.is_initialized = False

    def build_and_save(self, npz_directory, save_directory, batch_size=5):
        print("Start building the index...")
        start_time = time.time()
        
        os.makedirs(save_directory, exist_ok=True)
        
        npz_files = sorted(glob.glob(f"{npz_directory}/*.npz")) 
        total_files = len(npz_files)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        current_global_index = 0
        
        for i in range(0, total_files, batch_size):
            batch_files = npz_files[i:i+batch_size]
            batch_embeddings = []

            print(f"\nProcessing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1}")

            for file_idx, npz_file in enumerate(tqdm(batch_files, desc="Loading files")):
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    num_vectors = len(data['embeddings'])
                    
                    file_info = {
                        'file_path': npz_file,
                        'start_idx': current_global_index,
                        'end_idx': current_global_index + num_vectors,
                        'local_to_global': {j: current_global_index + j for j in range(num_vectors)}
                    }
                    self.file_mappings.append(file_info)
                    
                    current_global_index += num_vectors
                    
                    batch_embeddings.append(data['embeddings'])
                    self.descriptions.extend(data['descriptions'])
                    self.indices.extend([{
                        'file_idx': len(self.file_mappings) - 1,
                        'local_idx': idx
                    } for idx in data['indices']])
                    
                    del data
                    gc.collect()
                except Exception as e:
                    print(f"Error processing file {npz_file}: {str(e)}")
                    continue
            
            if batch_embeddings:
                batch_data = np.vstack(batch_embeddings)
                self.index.add(batch_data.astype('float32'))
                del batch_embeddings
                del batch_data
                gc.collect()
            
            print(f"The number of vectors currently processed: {self.index.ntotal}")

        index_path = os.path.join(save_directory, "faiss_index.bin")
        metadata_path = os.path.join(save_directory, "metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'descriptions': self.descriptions,
                'indices': self.indices,
                'file_mappings': self.file_mappings,
                'dimension': self.dimension
            }, f)
        
        total_time = time.time() - start_time
        print(f"\nIndex building completed!")
        print(f"Total number of vectors: {self.index.ntotal}")
        print(f"Number of processed files: {len(self.file_mappings)}")
        print(f"Building time: {total_time:.2f} seconds")
        print(f"Index has been saved to: {save_directory}")
        
        
        self.is_initialized = True
        return total_time

    def load_index(self, save_directory):
        print("Loading index and metadata...")
        start_time = time.time()
        
        index_path = os.path.join(save_directory, "faiss_index.bin")
        metadata_path = os.path.join(save_directory, "metadata.pkl")
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.descriptions = metadata['descriptions']
            self.indices = metadata['indices']
            self.file_mappings = metadata['file_mappings']
            self.dimension = metadata['dimension']
        
        load_time = time.time() - start_time
        print(f"Loading completed! Time taken: {load_time:.2f} seconds")
        print(f"Total number of vectors: {self.index.ntotal}")
        print(f"Number of loaded files: {len(self.file_mappings)}")

        self.is_initialized = True
        return load_time

    def to_gpu(self):
        if not self.is_initialized:
            raise RuntimeError("Please load the index first")

        print("\nStarting to transfer index to GPU...")
        start_time = time.time()
        
        try:
            self.res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
            print("Index has been successfully transferred to GPU")
        except Exception as e:
            print(f"Failed to transfer to GPU: {str(e)}")
            print("Continuing with CPU index...")
            self.gpu_index = None
        
        transfer_time = time.time() - start_time
        print(f"Transfer time: {transfer_time:.2f} seconds")
        return transfer_time
    
    def search(self, query_vector, k=10000):
        """Execute search to retrieve top k results, choosing between GPU or CPU based on k value"""
        if not self.is_initialized:
            raise RuntimeError("Please load the index first")

        start_time = time.time()
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        max_gpu_k = 2048
        
        if self.gpu_index is not None and k <= max_gpu_k:
            print(f"\nUsing GPU to query {k} results...")
            distances, indices = self.gpu_index.search(query_vector.astype('float32'), k)
        else:
            if k > max_gpu_k:
                print(f"\nk value ({k}) exceeds GPU limit ({max_gpu_k}), switching to CPU query...")
            else:
                print("\nUsing CPU to query...")

            # If GPU index exists, release it first
            if hasattr(self, 'gpu_index') and self.gpu_index is not None:
                print("Releasing GPU resources...")
                del self.gpu_index
                self.gpu_index = None
                gc.collect()  # Trigger garbage collection

            print("Executing CPU query...")
            distances, indices = self.index.search(query_vector.astype('float32'), k)

            # If needed, recreate GPU index after query
            if hasattr(self, 'res') and self.res is not None:
                print("Recreating GPU index...")
                self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        search_time = time.time() - start_time
        
        results = []
        for i, idx in enumerate(indices):
            if idx < len(self.indices):
                index_info = self.indices[idx]
                file_info = self.file_mappings[index_info['file_idx']]
                
                results.append({
                    'file_path': file_info['file_path'],
                    'local_index': int(index_info['local_idx']),
                    'global_index': int(idx),
                    'description': self.descriptions[idx],
                    'distance': float(distances[i])
                })

        print(f"\nQuery completed:")
        print(f"Query time: {search_time:.4f} seconds")
        print(f"Requested results: {k}")
        print(f"Actual results found: {len(results)}")

        return results, search_time

    def search_within_file(self, query_vector, target_file_path, k=100):
        """Search for nearest neighbor vectors only in the specified file"""
        if not self.is_initialized:
            raise RuntimeError("Please load the index first")

        # Find the mapping information for the target file
        target_file_info = None
        for file_info in self.file_mappings:
            if file_info['file_path'] == target_file_path:
                target_file_info = file_info
                break

        if target_file_info is None:
            raise ValueError(f"File not found: {target_file_path}")

        # Load the vectors for the target file
        file_data = np.load(target_file_path, allow_pickle=True)
        file_vectors = file_data['embeddings']

        # Create a temporary index containing only the target file's vectors
        temp_index = faiss.IndexFlatL2(self.dimension)
        temp_index.add(file_vectors.astype('float32'))

        # Prepare query vector
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Search within the file
        k = min(k, len(file_vectors))  # Ensure k does not exceed the number of vectors in the file
        distances, indices = temp_index.search(query_vector.astype('float32'), k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'file_path': target_file_path,
                'index': int(file_data['indices'][idx]),
                'local_index': int(idx),
                'description': file_data['descriptions'][idx],
                'distance': float(dist)
            })

        return results

    def get_file_info(self, global_index):
        """Get file information based on global index"""
        for file_info in self.file_mappings:
            if file_info['start_idx'] <= global_index < file_info['end_idx']:
                local_index = global_index - file_info['start_idx']
                return {
                    'file_path': file_info['file_path'],
                    'local_index': local_index,
                    'global_index': global_index
                }
        return None