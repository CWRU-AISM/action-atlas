# app/utils/openai_service.py
import sklearn.metrics.pairwise
from openai import OpenAI
import numpy as np
from .errors import APIError
from ..config import OPENAI_API_KEY, OPENAI_BASE_URL, MAX_RETRIES

class OpenAIService:
    _instance = None
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        self.max_retries = MAX_RETRIES
        
        self.query_optimization_system_prompt = """
        You are a professional query optimization assistant. Transform user input into a precise retrieval statement using EXACTLY this format:

        "references to '[topic]' and its associations with [related elements] across [scope/domain]"

        Rules:
        1. Extract ONE core topic (single word or simple phrase) from user input, even if input is a long sentence
        2. [related elements] must be 2-4 single words or simple phrases, separated by commas
        3. [scope/domain] must be 1-2 single words or simple phrases describing relevant fields
        4. Use the EXACT sentence structure - no variations allowed
        5. Output ONLY the formatted sentence, nothing else

        Your output will be used directly in retrieval systems.
        """
        self.query_optimization_user_prompt_template = """
        Transform this input into the required format:

        Examples:
        Input: "plant"
        Output: references to 'plant' and its associations with cultivation, agriculture, botany across biological sciences

        Input: "How do medical terms affect patient understanding?"
        Output: references to 'medical terminology' and its associations with patient communication, healthcare literacy, clinical practice across medical education

        Input: "machine learning algorithms for image processing"
        Output: references to 'machine learning' and its associations with algorithms, image processing, computer vision across artificial intelligence

        Input: {}
        Output: """

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = OpenAIService()
        return cls._instance

    def get_embedding(self, text, model="text-embedding-3-large"):
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise APIError('EMBEDDING_ERROR', f"EMBEDDING_ERROR: {str(e)}")
                continue
            
    def generate_optimized_query(self, vague_term, model="gpt-4o"):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.query_optimization_system_prompt},
                        {"role": "user", "content": self.query_optimization_user_prompt_template.format(vague_term)}
                    ],
                    max_tokens=100,
                    temperature=0.2,
                )
                print(response.choices[0].message.content.strip())
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise APIError('QUERY_OPTIMIZATION_ERROR', f"QUERY_OPTIMIZATION_ERROR: {str(e)}")
                continue

if __name__ == "__main__":
    openai_service = OpenAIService().get_instance()
    
    vague_term = "felines"
    optimized_query = openai_service.generate_optimized_query(vague_term)
    print(f"Vague term: {vague_term}")
    print(f"Optimized query: {optimized_query}")