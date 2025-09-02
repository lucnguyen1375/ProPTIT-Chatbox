from docx import Document
from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
import openai
import os



doc = Document("CLB_PROPTIT.docx")

# Sửa chỗ FIX_ME để dùng DB mà các em muốn hoặc các em có thể tự sửa code trong lớp VectorDatabase để dùng các DB khác

vector_db = VectorDatabase(db_type= 'mongodb') # baseline model: "mongodb"

# Sửa chỗ FIX_ME để dùng embedding model mà các em muốn, hoặc các em có tự thêm embedding model trong lớp Embeddings

embedding = Embeddings(model_name='gemini-embedding-001', type= 'gemini') 



# TODO: Embedding từng document trong file CLB_PROPTIT.docx và lưu vào DB. 
# Code dưới là sử dụng mongodb, các em có thể tự sửa lại cho phù hợp với DB mà mình đang dùng
#--------------------Code Lưu Embedding Document vào DB--------------------------
cnt = 1
if vector_db.count_documents("information") == 0:
    for para in doc.paragraphs:
        if para.text.strip():
            embedding_vector = embedding.encode(para.text)
            # Lưu vào cơ sở dữ liệu
            vector_db.insert_document(
                collection_name="information",
                document={
                    "title": f"Document {cnt}",
                    "information": para.text,
                    "embedding": embedding_vector
                }
            )
            cnt += 1
else:
    print("Documents already exist in the database. Skipping insertion.")
    

#------------------------------------------------------------------------------------

from metrics_rag import *

df_retrieval_metrics = calculate_metrics_retrieval("CLB_PROPTIT.csv", "test_data_proptit.xlsx", embedding, vector_db, False) # đặt là True nếu là tập train, False là tập test
df_llm_metrics = calculate_metrics_llm_answer("CLB_PROPTIT.csv", "test_data_proptit.xlsx", embedding, vector_db, False) # đặt là True nếu là tập train, False là tập test
print(df_retrieval_metrics.head())
print(df_llm_metrics.head())








