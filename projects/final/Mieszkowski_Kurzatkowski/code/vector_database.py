import faiss
import json
import numpy as np


class VectorDatabaseWraper:
    def __init__ (self, folder_path = None, embedding_size = 1536):
        if folder_path:
            self.index = faiss.read_index(f"{folder_path}/database.index")
            with open(f"{folder_path}/metadata.json", 'r') as infile:
                self.metadata = json.load(infile)
        else:
            self.index = faiss.IndexFlatL2(embedding_size)
            self.metadata = []

    def save(self, folder_path):
        faiss.write_index(self.index, f"{folder_path}/database.index")
        with open(f"{folder_path}/metadata.json", 'w') as outfile:
            json.dump(self.metadata, outfile)

    def add(self, embeddings, metadata):
        if len(embeddings) == 0:
            return
        self.index.add(np.array(embeddings))
        self.metadata.extend(metadata)

    def search(self, query_vector, k):
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        print(indices[0])
        print(distances)
        print([self.metadata[i] for i in indices[0]])
        return [self.metadata[i] for i in indices[0]], distances[0]
    
    def has_record(self, metadata):
        return metadata in self.metadata


 
        
    