# from pymilvus import MilvusClient, DataType
# import numpy as np
# import concurrent.futures
# import os

# class MilvusManager:
#     def __init__(self, milvus_uri, collection_name, create_collection, dim=128):
#         self.client = MilvusClient(uri=milvus_uri)
#         self.collection_name = collection_name
#         self.dim = dim
#         if self.client.has_collection(collection_name=self.collection_name):
#             print(f"Collection {collection_name} already exists. Loading {collection_name}")
#             self.client.load_collection(collection_name)
#         else:
#             if create_collection:
#                 self.create_collection()
#                 self.create_index()
            
            

#     # def connect(self, milvus_uri, collection_name, create_collection, dim):
#     #     try:
#     #         self.client = MilvusClient(uri=milvus_uri)
#     #         self.collection_name = collection_name
#     #         self.dim = dim
#     #         if self.client.has_collection(collection_name=self.collection_name):
#     #             print(f"Collection {collection_name} already exists. Loading {collection_name}")
#     #             self.client.load_collection(collection_name)
#     #         else:
#     #             if create_collection:
#     #                 self.create_collection()
#     #                 self.create_index()
#     #         self.connected = True
#     #         return True
#     #     except Exception as e:
#     #         print(f"Error connecting to Milvus: {e}")
#     #         self.connected = False
#     #         return False
    
#     def disconnect(self):
#         if hasattr(self, "client") and self.client:
#             try:
#                 self.client.close()
#                 print("Disconnected from Milvus.")
#             except Exception as e:
#                 print(f"Error disconnecting from Milvus: {e}")


#     def create_collection(self):
#         if self.client.has_collection(collection_name=self.collection_name):
#             self.client.drop_collection(collection_name=self.collection_name)
#         schema = self.client.create_schema(
#             auto_id=True,
#             enable_dynamic_fields=True,
#         )
#         schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
#         schema.add_field(
#             field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
#         )
#         schema.add_field(field_name="seq_id", datatype=DataType.INT16)
#         schema.add_field(field_name="doc_id", datatype=DataType.INT64)
#         schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

#         self.client.create_collection(
#             collection_name=self.collection_name, schema=schema
#         )

#     def create_index(self):
#         self.client.release_collection(collection_name=self.collection_name)
#         self.client.drop_index(
#             collection_name=self.collection_name, index_name="vector"
#         )
#         index_params = self.client.prepare_index_params()
#         index_params.add_index(
#             field_name="vector",
#             index_name="vector_index",
#             index_type="HNSW", 
#             metric_type="IP", 
#             params={
#                 "M": 16,
#                 "efConstruction": 500,
#             },
#         )

#         self.client.create_index(
#             collection_name=self.collection_name, index_params=index_params, sync=True
#         )

#     def create_scalar_index(self):
#         self.client.release_collection(collection_name=self.collection_name)

#         index_params = self.client.prepare_index_params()
#         index_params.add_index(
#             field_name="doc_id",
#             index_name="int32_index",
#             index_type="INVERTED",
#         )

#         self.client.create_index(
#             collection_name=self.collection_name, index_params=index_params, sync=True
#         )

#     def search(self, data, topk):
#         search_params = {"metric_type": "IP", "params": {}}
#         results = self.client.search(
#             self.collection_name,
#             data,
#             limit=int(50),
#             output_fields=["vector", "seq_id", "doc_id", "doc"],
#             search_params=search_params,
#         )
#         doc_ids = set()
#         doc_paths = {}
#         for r_id in range(len(results)):
#             for r in range(len(results[r_id])):
#                 doc_id = results[r_id][r]["entity"]["doc_id"]
#                 doc_ids.add(doc_id)
#                 doc_paths[doc_id] = results[r_id][r]["entity"]["doc"]

#         scores = []

#         def rerank_single_doc(doc_id, data, client, collection_name):
#             doc_colbert_vecs = client.query(
#                 collection_name=collection_name,
#                 filter=f"doc_id in [{doc_id}, {doc_id + 1}]",
#                 output_fields=["seq_id", "vector", "doc"],
#                 limit=1000,
#             )
#             doc_vecs = np.vstack(
#                 [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
#             )
#             score = np.dot(data, doc_vecs.T).max(1).sum()
#             return (score, doc_id)

#         with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
#             futures = {
#                 executor.submit(
#                     rerank_single_doc, doc_id, data, self.client, self.collection_name
#                 ): doc_id
#                 for doc_id in doc_ids
#             }
#             for future in concurrent.futures.as_completed(futures):
#                 score, doc_id = future.result()
#                 scores.append((score, doc_id, doc_paths[doc_id]))

#         # self.disconnect()

#         scores.sort(key=lambda x: x[0], reverse=True)
#         if len(scores) >= topk:
#             return scores[:topk]
#         else:
#             return scores

#     def insert(self, data):
#         colbert_vecs = [vec for vec in data["colbert_vecs"]]
#         seq_length = len(colbert_vecs)
#         doc_ids = [data["doc_id"] for i in range(seq_length)]
#         seq_ids = list(range(seq_length))

#         # Assign the correct "doc" values for every entry
#         docs = [f"{os.path.dirname(data['filepath'])}" for i in range(seq_length)]

#         self.client.insert(
#             self.collection_name,
#             [
#                 {
#                     "vector": colbert_vecs[i],
#                     "seq_id": seq_ids[i],
#                     "doc_id": doc_ids[i],
#                     "doc": docs[i],
#                 }
#                 for i in range(seq_length)
#             ],
#         )


#     def get_images_as_doc(self, images_with_vectors:list):
        
#         images_data = []

#         for i in range(len(images_with_vectors)):
#             data = {
#                 "colbert_vecs": images_with_vectors[i]["colbert_vecs"],
#                 "doc_id": i,
#                 "filepath": images_with_vectors[i]["filepath"],
#             }
#             images_data.append(data)

#         return images_data


#     def insert_images_data(self, image_data):
#         data = self.get_images_as_doc(image_data)

#         for i in range(len(data)):
#             self.insert(data[i])

#     def get_indexed_file_names(self):
#         """
#         Retrieve all indexed file names from Milvus.
#         """
        
#         # Retrieve only the 'doc' field from the database (which stores file paths)
#         results = self.client.query(
#             collection_name=self.collection_name,
#             filter="doc != ''",  # Retrieve all non-empty file paths
#             output_fields=["doc"],
#             limit=10000  # Adjust limit as needed
#         )

#         # Extract unique file names (subdirectory names) from file paths
#         file_names = set(item["doc"].split('//')[0] for item in results if "doc" in item)

#         return list(file_names)

# =========================================================





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
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=50,
            output_fields=["vector", "seq_id", "doc_id", "doc"],
            search_params=search_params,
        )

        # Instead of grouping by doc_id only, we now use the tuple (doc_id, doc)
        doc_identifiers = set()
        doc_paths = {}  # Maps (doc_id, doc) -> doc (or file name/path)
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                entity = results[r_id][r]["entity"]
                doc_id = entity["doc_id"]
                doc = entity["doc"]
                identifier = (doc_id, doc)
                doc_identifiers.add(identifier)
                doc_paths[identifier] = doc

        scores = []

        def rerank_single_doc(doc_id, doc, query_vecs, client, collection_name):
            # Use both doc_id and doc in the filter to ensure that pages are fetched from the correct PDF
            filter_str = f"doc_id == {doc_id} AND doc == '{doc}'"
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=filter_str,
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            # Stack all the vectors belonging to this document.
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            # Compute a score: for each query vector, take the max dot product over all page vectors and sum them up.
            score = np.dot(query_vecs, doc_vecs.T).max(axis=1).sum()
            return (score, doc_id, doc)

        # Use a thread pool to rerank each document in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, doc, data, self.client, self.collection_name
                ): (doc_id, doc)
                for (doc_id, doc) in doc_identifiers
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id, doc = future.result()
                scores.append((score, doc_id, doc_paths[(doc_id, doc)]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:topk] if len(scores) >= topk else scores

    def insert(self, data):
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        # Here, all pages from the same PDF have the same doc_id.
        doc_ids = [data["doc_id"] for _ in range(seq_length)]
        seq_ids = list(range(seq_length))
        # Use the file name (or its directory) as the document identifier.
        docs = [f"{os.path.dirname(data['filepath'])}" for _ in range(seq_length)]

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

    def get_images_as_doc(self, images_with_vectors: list):
        images_data = []
        for i in range(len(images_with_vectors)):
            data = {
                "colbert_vecs": images_with_vectors[i]["colbert_vecs"],
                # Ensure that each PDF gets a unique doc_id.
                "doc_id": i,
                "filepath": images_with_vectors[i]["filepath"],
            }
            images_data.append(data)
        return images_data

    def insert_images_data(self, image_data):
        data = self.get_images_as_doc(image_data)
        for item in data:
            self.insert(item)

    def get_indexed_file_names(self):
        """
        Retrieve all indexed file names from Milvus.
        """
        results = self.client.query(
            collection_name=self.collection_name,
            filter="doc != ''",  # Retrieve all non-empty file paths
            output_fields=["doc"],
            limit=10000  # Adjust limit as needed
        )
        file_names = set(item["doc"].split('//')[0] for item in results if "doc" in item)
        return list(file_names)
