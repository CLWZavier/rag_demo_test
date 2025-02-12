from colpali_manager import ColpaliManager
from milvus_manager import MilvusManager
from pdf_manager import PdfManager
import hashlib
import os

pdf_manager = PdfManager()
colpali_manager = ColpaliManager()



class Middleware:
    def __init__(self, id:str = None, create_collection=True):
        """
        Constructor for Middleware class.

        Parameters:
        id (str): unique id for the pdf to be indexed.
        create_collection (bool, optional): whether to create a new collection in the Milvus database. Defaults to True.

        """
        # hashed_id = hashlib.md5(id.encode()).hexdigest()[:8]
        # milvus_db_name = f"milvus_{hashed_id}.db"
        milvus_db_name = "tcp://localhost:19530"
        self.milvus_manager = MilvusManager(milvus_db_name, "colpali", create_collection)

    def index(self, pdf_path: str, id:str, max_pages: int, pages: list[int] = None):
        
        print(f"Indexing {pdf_path}, id: {id}, max_pages: {max_pages}")

        image_paths = pdf_manager.save_images(id, pdf_path, max_pages)

        print(f"Saved {len(image_paths)} images")

        colbert_vecs = colpali_manager.process_images(image_paths)

        images_data = [{
            "colbert_vecs": colbert_vecs[i],
            "filepath": image_paths[i]
        } for i in range(len(image_paths))]

        print(f"Inserting {len(images_data)} images data to Milvus")

        self.milvus_manager.insert_images_data(images_data)

        print("Indexing completed")

        return image_paths

        
    def search(self, search_queries: list[str], num_results):
        print(f"Searching for {len(search_queries)} queries")

        final_res = []

        for query in search_queries:
            print(f"Searching for query: {query}")
            query_vec = colpali_manager.process_text([query])[0]
            search_res = self.milvus_manager.search(query_vec, topk=num_results)
            print(f"Search result: {search_res} for query: {query}")
            final_res.append(search_res)

        return final_res

    def list_index(self):
        return self.milvus_manager.get_indexed_file_names()