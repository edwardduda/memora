from sentence_transformers import SentenceTransformer


class Memora:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str):
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def find(self):
        pass

    def compare(self):
        pass

    def insert(self):
        pass

    def remove(self):
        pass

    def normalize(self):
        pass

    def update(self):
        pass