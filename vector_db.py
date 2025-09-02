from pymongo import MongoClient
from chromadb import HttpClient
from qdrant_client import QdrantClient
from supabase import create_client, Client
from dotenv import load_dotenv
from qdrant_client import models as qdrant_models
load_dotenv()
import os


# Các em có thể tự thêm vector database mới hoặc dùng các database có sẵn
class VectorDatabase:
    def __init__(self, db_type: str):
        self.db_type = db_type
        if self.db_type == "mongodb":
            self.client = MongoClient(os.getenv("MONGODB_URI"))
        elif self.db_type == "chromadb":
            self.client = HttpClient(
                host="localhost", 
                port=8123
            )
        elif self.db_type == "qdrant":
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_KEY"),
            )
        elif self.db_type == "supabase":
            url: str = os.environ.get("SUPABASE_URL")
            key: str = os.environ.get("SUPABASE_KEY")
            supabase: Client = create_client(
                supabase_url=url,
                supabase_key=key
                )
            self.client = supabase
    def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists for Qdrant, create if it doesn't"""
        if self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Info] Collection '{collection_name}' not found. Creating it...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=3072,  # adjust size based on your embedding model
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                
                # Create index for title field to enable filtering
                print(f"[Info] Creating index for 'title' field...")
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="title",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                return True  # Collection was created
        return False  # Collection already existed or not Qdrant
    def insert_document(self, collection_name: str, document: dict):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            collection.insert_one(document)
        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name=collection_name)
            collection.add(
                documents=[document["information"]],
                embeddings=[document["embedding"]],
                ids=[document["title"]]
            )
        elif self.db_type == "qdrant":
            self._ensure_collection_exists(collection_name)
            
            # Insert the document as a point
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": hash(document["title"]) % (2**63),  # Generate unique ID from title
                        "vector": document["embedding"],
                        "payload": {
                            "title": document["title"],
                            "information": document["information"]
                        }
                    }
                ]
            )
        elif self.db_type == "supabase":
            self.client.table(collection_name).insert(document).execute()
    def query(self, collection_name: str, query_vector: list, limit: int = 5):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # tên index bạn đã tạo
                        "queryVector": query_vector,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": limit
                    }
                }
            ])
            return list(results)
        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )
            docs = []
            for i in range(len(results["ids"][0])):
                docs.append({
                    "title": results["ids"][0][i],
                    "information": results["documents"][0][i]
                })
            return docs
        elif self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Warning] Collection '{collection_name}' doesn't exist for querying")
                return []
                
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # Format results to match expected structure
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.payload["title"],
                    "information": result.payload["information"],
                    "score": result.score
                })
            return formatted_results
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*").execute()
            return response.data
    def document_exists(self, collection_name, filter_query):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            return collection.count_documents(filter_query) > 0
        elif self.db_type == "chromadb":
            try:
                collection = self.client.get_or_create_collection(name=collection_name)
                # Lấy toàn bộ ID hiện có trong collection
                all_ids = collection.get()["ids"]
                return filter_query["title"] in all_ids
            except Exception as e:
                print(f"Error checking existence in ChromaDB: {e}")
                return False
        elif self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Info] Collection '{collection_name}' doesn't exist yet")
                return False
                
            # Search for document with matching title
            try:
                result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [
                            {
                                "key": "title",
                                "match": {"value": filter_query["title"]}
                            }
                        ]
                    },
                    limit=1
                )
                return len(result[0]) > 0
            except Exception as e:
                print(f"Error checking document existence in Qdrant: {e}")
                return False
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*").eq("title", filter_query["title"]).execute()
            return len(response.data) > 0
        else:
            raise ValueError("Unsupported database type")
    def count_documents(self, collection_name: str) -> int:
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")  # Đảm bảo đúng tên DB
            collection = db[collection_name]
            return collection.count_documents({})
        else:
            raise NotImplementedError("count_documents chỉ hỗ trợ MongoDB trong phiên bản này.")
    def drop_collection(self, collection_name: str):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            collection.drop()
        elif self.db_type == "chromadb":
            self.client.delete_collection(name=collection_name)
        elif self.db_type == "qdrant":
            if self.client.collection_exists(collection_name=collection_name):
                self.client.delete_collection(collection_name=collection_name)
        elif self.db_type == "supabase":
            self.client.table(collection_name).delete().execute()
        else:
            raise ValueError("Unsupported database type for drop_collection")
        
# Hàm tính toán tất cả metrics liên quan đến Retrieval

def calculate_metrics_retrieval(file_clb_proptit, file_train , embedding, vector_db, train):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "hit@k": [],
        "recall@k": [],
        "precision@k": [],
        "f1@k": [],
        "map@k": [],
        "mrr@k": [],
        "ndcg@k": [],
        "context_precision@k": [],
        "context_recall@k": [],
        "context_entities_recall@k": []
    }
    
    print('CACULATING METRICS RETRIEVAL...')
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        print(f"Calculating metrics retrieval for k={k}...", end = ' ')
        metrics["K"].append(k)
        print("Creating embeddings and retrieving documents...")
        df_train=embedding_file(file_train, embedding, vector_db, k)
        print("Calculating hit@k...", end = ' ')
        metrics["hit@k"].append(round(hit_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["hit@k"][-1])
        print("Calculating recall@k...", end = ' ')
        metrics["recall@k"].append(round(recall_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["recall@k"][-1])
        print("Calculating precision@k...", end = ' ')
        metrics["precision@k"].append(round(precision_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["precision@k"][-1])
        print("Calculating f1@k...", end = ' ')
        metrics["f1@k"].append(round(f1_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["f1@k"][-1])
        print("Calculating map@k...", end = ' ')
        metrics["map@k"].append(round(map_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["map@k"][-1])
        print("Calculating mrr@k...", end = ' ')
        metrics["mrr@k"].append(round(mrr_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["mrr@k"][-1])
        print("Calculating ndcg@k...", end = ' ')
        metrics["ndcg@k"].append(round(ndcg_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["ndcg@k"][-1])
        print("Calculating context_precision@k...", end = ' ')
        metrics["context_precision@k"].append(round(context_precision_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["context_precision@k"][-1])
        print("Calculating context_recall@k...", end = ' ')
        metrics["context_recall@k"].append(round(context_recall_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["context_recall@k"][-1])
        print("Calculating context_entities_recall@k...", end = ' ')
        metrics["context_entities_recall@k"].append(round(context_entities_recall_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(metrics["context_entities_recall@k"][-1])
    
    print('#' * 50)
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_retrieval_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_retrieval_test.csv", index=False)
    return metrics_df