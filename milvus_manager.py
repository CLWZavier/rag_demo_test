from pymilvus import MilvusClient, DataType
import numpy as np
import concurrent.futures
import os

class MilvusManager:
    def __init__(self, milvus_uri, collection_name, create_collection, dim=128):
        # self.client = MilvusClient(uri=milvus_uri)
        # self.collection_name = collection_name
        # if self.client.has_collection(collection_name=self.collection_name):
        #     self.client.load_collection(collection_name)
        # self.dim = dim

        # if create_collection:
        #     self.create_collection()
        #     self.create_index()
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.create_collection = create_collection
        self.dim = dim
        self.connected = False
        self.connect(milvus_uri, collection_name, create_collection, dim)

    def connect(self, milvus_uri, collection_name, create_collection, dim):
        try:
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
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
                print("Disconnected from Milvus.")
            except Exception as e:
                print(f"Error disconnecting from Milvus: {e}")


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
            metric_type="IP", 
            params={
                "M": 16,
                "efConstruction": 500,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        try:
            if not self.connected:
                self.connect(self.milvus_uri, self.collection_name, self.create_collection, self.dim)
        except Exception as e:
            print(f"Error reconnecting to Milvus for search: {e}")

        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(500),
            output_fields=["vector", "seq_id", "doc_id", "doc"],
            search_params=search_params,
        )
        doc_ids = set()
        doc_paths = {}
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                doc_id = results[r_id][r]["entity"]["doc_id"]
                doc_ids.add(doc_id)
                doc_paths[doc_id] = results[r_id][r]["entity"]["doc"]

        scores = []

        # In milvus_manager.py's rerank_single_doc function
        def rerank_single_doc(doc_id, data, client, collection_name):
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}, {doc_id + 1}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            dot_product = np.dot(data, doc_vecs.T)
            max_indices = np.argmax(dot_product, axis=1).tolist()
            score = np.max(dot_product, axis=1).sum()
            return (score, doc_id, max_indices)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self.client, self.collection_name
                ): doc_id
                for doc_id in doc_ids
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id, max_indices = future.result()
                scores.append((score, doc_id, doc_paths[doc_id], max_indices))

        # self.disconnect()

        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        try:
            if not self.connected:
                self.connect(self.milvus_uri, self.collection_name, self.create_collection, self.dim)
        except Exception as e:
            print(f"Error reconnecting to Milvus for search: {e}")

        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))

        # Assign the correct "doc" values for every entry
        docs = [f"{os.path.dirname(data['filepath'])}" for i in range(seq_length)]

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
        try:
            if not self.connected:
                self.connect(self.milvus_uri, self.collection_name, self.create_collection, self.dim)
        except Exception as e:
            print(f"Error reconnecting to Milvus for search: {e}")
        data = self.get_images_as_doc(image_data)

        for i in range(len(data)):
            self.insert(data[i])

    def get_indexed_file_names(self):
        """
        Retrieve all indexed file names from Milvus.
        """
        try:
            if not self.connected:
                self.connect(self.milvus_uri, self.collection_name, self.create_collection, self.dim)
        except Exception as e:
            print(f"Error reconnecting to Milvus for search: {e}")
        
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