from pymilvus import MilvusClient, DataType
import numpy as np
import concurrent.futures
import os

class MilvusManager:
    def __init__(self, milvus_uri, collection_name, create_collection, dim=128):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.dim = dim
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Collection {collection_name} already exists. Loading {collection_name}")
            self.client.load_collection(collection_name)
        else:
            if create_collection:
                self.create_collection()
                self.create_index()

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="COSINE",  # Changed to COSINE for better semantic matching
            params={
                "M": 48,
                "efConstruction": 800,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 400,
                "nprobe": 16
            }
        }
        
        # Initial search with higher limit to get a good candidate pool
        results = self.client.search(
            self.collection_name,
            data,
            limit=topk,
            output_fields=["vector", "seq_id", "doc_id", "doc"],
            search_params=search_params,
        )

        print(f"len of results = {len(results)}")

        # Collect unique doc_ids and their paths
        doc_ids = set()
        doc_paths = {}
        for r_id in range(len(results)):
            for r in results[r_id]:
                doc_id = r["entity"]["doc_id"]
                doc_ids.add(doc_id)
                doc_paths[doc_id] = r["entity"]["doc"]
        
        print(f"doc_ids = {doc_ids}")
        print(f"doc_paths = {doc_paths}")

        # Rerank across all documents
        scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(doc_ids), 32)) as executor:
            futures = []
            for doc_id in doc_ids:
                future = executor.submit(
                    self._rerank_doc, 
                    doc_id=doc_id,
                    query_vector=data,
                    doc_path=doc_paths[doc_id]
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    score, doc_id, doc_path = future.result()
                    scores.append((score, doc_id, doc_path))
                except Exception as e:
                    print(f"Error in reranking: {e}")
                    continue

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:topk]

    def _rerank_doc(self, doc_id, query_vector, doc_path):
        """Rerank a single document's pages against the query"""
        # Get vectors for this document
        doc_vectors = self.client.query(
            collection_name=self.collection_name,
            filter=f"doc_id == {doc_id}",
            output_fields=["vector"],
            limit=10
        )
        
        if not doc_vectors:
            return (0, doc_id, doc_path)

        # Stack vectors and compute similarities
        doc_vecs = np.vstack([d["vector"] for d in doc_vectors])
        similarities = np.dot(query_vector, doc_vecs.T)
        
        # Take max similarity as the score
        score = similarities.max(1).mean()
        
        return score, doc_id, doc_path

    def insert(self, data):
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        
        # Normalize vectors for cosine similarity
        colbert_vecs = [vec / np.linalg.norm(vec) for vec in colbert_vecs]
        
        doc_ids = [data["doc_id"] for _ in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [os.path.dirname(data["filepath"]) for _ in range(seq_length)]

        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "seq_id": seq_ids[i],
                    "doc_id": doc_ids[i],
                    "doc": docs[i],
                }
                for i in range(seq_length)
            ],
        )

    def get_images_as_doc(self, images_with_vectors:list):
        
        images_data = []

        for i in range(len(images_with_vectors)):
            data = {
                "colbert_vecs": images_with_vectors[i]["colbert_vecs"],
                "doc_id": i,
                "filepath": images_with_vectors[i]["filepath"],
            }
            images_data.append(data)

        return images_data


    def insert_images_data(self, image_data):
        # try:
        #     if not self.connected:
        #         self.connect(self.milvus_uri, self.collection_name, self.create_collection, self.dim)
        # except Exception as e:
        #     print(f"Error reconnecting to Milvus for search: {e}")
        data = self.get_images_as_doc(image_data)

        for i in range(len(data)):
            self.insert(data[i])

    def get_indexed_file_names(self):
        """
        Retrieve all indexed file names from Milvus.
        """
        # try:
        #     if not self.connected:
        #         self.connect(self.milvus_uri, self.collection_name, self.create_collection, self.dim)
        # except Exception as e:
        #     print(f"Error reconnecting to Milvus for search: {e}")
        
        # Retrieve only the 'doc' field from the database (which stores file paths)
        results = self.client.query(
            collection_name=self.collection_name,
            filter="doc != ''",  # Retrieve all non-empty file paths
            output_fields=["doc"],
            limit=10000  # Adjust limit as needed
        )

        # Extract unique file names (subdirectory names) from file paths
        file_names = set(item["doc"].split('//')[0] for item in results if "doc" in item)

        return list(file_names)