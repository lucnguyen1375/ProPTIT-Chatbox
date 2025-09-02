import os
from dotenv import load_dotenv
load_dotenv()

from time import sleep
import pandas as pd
from docx import Document
from embeddings import Embeddings
from vector_db import VectorDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google import genai
from google.genai import types
from rerank import Reranker
from tqdm import tqdm

reranker = Reranker(model_name="namdp-ptit/ViRanker", use_fp16=True, normalize=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_transform(query):
    messages = [
        {
            "role": "system",
            "content": """Bạn là một trợ lý AI giúp cải thiện các truy vấn tìm kiếm để truy xuất thông tin tốt hơn.
Hãy biến đổi truy vấn sau thành một truy vấn tốt hơn, rõ ràng hơn.
Nếu câu văn không đủ tốt, hãy cải thiện nó.
Yêu cầu bắt buộc: 
- Giữ nguyên ý nghĩa ban đầu của truy vấn.
- Không thêm thông tin mới.
- Chỉ trả về truy vấn đã được cải thiện, không có lời giải thích hay thông tin bổ sung nào khác."""
        },
        {
            "role": "user",
            "content": f"Truy vấn gốc: {query}"
        }
    ]
        
    reply = client.chat.completions.create(
        model="gpt-4o-mini",  # Hoặc "gpt-3.5-turbo", "gpt-4", "gpt-4o"
        messages = messages,   
    ).choices[0].message.content.strip()
    return reply


def embedding_file(file_train_data_proptit, embedding, vector_db, k):
    df_train = pd.read_excel(file_train_data_proptit)
    result_list = []
    df_train['User embedding'] = None
    df_train['Information'] = None
    for index, row in df_train.iterrows():
        raw_query = row['Query']
        query = query_transform(raw_query)
        ground_truth_doc = row['Ground truth document']
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = embedding.encode(query)

        df_train.at[index, 'User embedding'] = user_embedding
        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = vector_db.query("information", user_embedding, limit = k + 5) # Lấy nhiều hơn k để rerank
        
        # retrieved_docs = [int(result['title'].split()[-1]) for result in results if 'title' in result]    
        retrieved_docs = []
        
        passages = [result["information"] for result in results]
        scores, ranked_passages = reranker(query, passages)
        
        
        # Lấy danh sách tài liệu được truy suất
        for result in results:
            if result['information'] in ranked_passages[:k]:
                retrieved_docs.append(int(result['title'].split()[-1]))
            
        df_train.at[index, 'Information'] = ranked_passages[:k]
        result_list.append(retrieved_docs)
        
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        
        df_train.at[index, 'Ground truth document'] = ground_truth_docs
        
        # print('----------------------------------')
        # print(f"Index: {index}")
        # print(f"Query: {query}")
        # print(f"Ground truth document: {ground_truth_doc}")
        # print(f"Retrieved documents: {retrieved_docs}")
        
        
    df_train['Retrieved documents'] = result_list    
    # print(df_train.head())
    for index, row in df_train.head().iterrows():
        print(row['Retrieved documents'])
        for doc in row['Information']:
            print(doc)
    
    sleep(100)
    return df_train

def hit_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    hits = 0
    total_queries = len(df_train)

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_docs = row['Ground truth document']
        retrieved_docs = row['Retrieved documents']
        
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits += 1
            
    return hits / total_queries if total_queries > 0 else 0



# Hàm recall@k
def recall_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_docs = row['Ground truth document']
        retrieved_docs = row['Retrieved documents']
        
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in ground_truth_docs if doc in retrieved_docs])
        ans += hits / len(ground_truth_docs) 
        
    # print('✅ Recall_k metrics : Done')
    return ans / len(df_train)

# Hàm precision@k
def precision_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_docs = row['Ground truth document']
        retrieved_docs = row['Retrieved documents']
        
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in retrieved_docs if doc in ground_truth_docs])
        ans += hits / k 
        # print("Hits / k for this query:", hits / k)
    return ans / len(df_train)


# Hàm f1@k
def f1_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    precision = precision_k(file_clb_proptit, df_train, embedding, vector_db, k)
    recall = recall_k(file_clb_proptit, df_train, embedding, vector_db, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Hàm MAP@k

def map_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    total_map = 0

    for index, row in df_train.iterrows():
        hits = 0
        ap = 0
        query = row['Query']
        ground_truth_docs = row['Ground truth document']
        retrieved_docs = row['Retrieved documents']
        
        # Tính MAP cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                hits += 1
                ap += hits / (i + 1)
        if hits > 0:
            ap /= hits
        # print(f"Average Precision for this query: {ap}")
        total_map += ap 
    return total_map / len(df_train)

# Hàm MRR@k
def mrr_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    total_mrr = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_docs = row['Ground truth document']
        retrieved_docs = row['Retrieved documents']
        
        # Tính MRR cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                total_mrr += 1 / (i + 1)
                break
    return total_mrr / len(df_train) if len(df_train) > 0 else 0

# Hàm NDCG@k
import numpy as np
def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

def ndcg_at_k(relevances, k):
    dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(relevances, k) / dcg_max

def similarity(embedding1, embedding2):
    # Giả sử ta có một hàm để tính độ tương đồng giữa hai embedding
    # Ở đây ta sẽ sử dụng cosine similarity, chuẩn hóa để score nước về khoảng [0, 1]
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = dot_product / (norm1 * norm2)
    return (cos_sim + 1) / 2


def ndcg_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    total_ndcg = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_docs = row['Ground truth document']
        retrieved_docs = row['Retrieved documents']


        # Nếu điểm tương đồng > 0.9 thì gán 3, nếu > 0.7 thì gán 2, nếu > 0.5 thì gán 1, còn lại thì gán 0 
        relevances = []
        for doc_index, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                # Giả sử ta có một hàm để tính độ tương đồng giữa câu hỏi và tài liệu, doc là số thứ tự của tài liệu trong file CLB_PROPTIT.csv
                user_embedding = df_train.at[doc_index, 'User embedding']
                similarity_score = similarity(user_embedding, embedding.encode(df_clb.loc[doc-1, 'Văn bản']))
                if similarity_score > 0.9:
                    relevances.append(3)
                elif similarity_score > 0.7:
                    relevances.append(2)
                elif similarity_score > 0.5:
                    relevances.append(1)
                else:
                    relevances.append(0)
            else:
                relevances.append(0)
        ndcg = ndcg_at_k(relevances, k)
        # print(f"NDCG for this query: {ndcg}")
        total_ndcg += ndcg

    return total_ndcg / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Precision@k (LLM Judged)
def context_precision_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    total_precision = 0

    for index, row in df_train.iterrows():
        # TODO: Tạo ra LLM Answer, các em hãy tự viết phần system prompt
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        results = row['Retrieved documents']
        information_texts = row['Information']
        
        # TODO: viết câu query của người dùng (bao gồm document retrieval và câu query)
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([information_text for information_text in information_texts])

        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi của người dùng: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        reply = response.choices[0].message.content.strip()

        # Đẩy các đoạn văn được retrieved và câu trả lời của LLM vào một LLM Judged context với prompt system
        # LLM Judged context
        for information_text in information_texts:
            # NOTE: Các em có thể chỉnh messages_judged nếu muốn
            messages_judged = [
                {
                    "role": "system",
                    "content": """Bạn là một trợ lý AI chuyên đánh giá độ chính xác của các câu trả lời dựa trên ngữ cảnh được cung cấp. Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trả lời từ một mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời dựa trên ngữ cảnh và câu hỏi. Nếu ngữ cảnh và câu hỏi cung cấp đủ thông tin hoặc chỉ cần một phần thông tin để trả lời câu hỏi, hãy đánh giá câu trả lời là 1. Nếu không, hãy đánh giá là 0. Hãy đọc thật kĩ ngữ cảnh, chỉ cần ngữ cảnh có một phần thông tin để trả lời cho một phần của câu hỏi thì cũng đánh giá là 1. Nếu ngữ cảnh không liên quan gì đến câu hỏi, hãy đánh giá là 0. LƯU Ý: Chỉ trả lời 1 hoặc 0, không giải thích gì thêm."""
                }
            ]
            # TODO: "content" sẽ lưu ngữ cảnh, câu hỏi, câu trả lời
            messages_judged.append({
                "role": "user",
                "content": f"Ngữ cảnh: {information_text}\n\nCâu hỏi: {query}\n\nCâu trả lời: {reply}"
            })
            # Gọi API đến LLM Judged
            judged_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_judged
            )
            judged_reply = judged_response.choices[0].message.content.strip()
            if judged_reply == "1":
                hits += 1
        precision = hits / k if k > 0 else 0
        total_precision += precision
    return total_precision / len(df_train) if len(df_train) > 0 else 0


# Hàm Context Recall@k (LLM Judged)
def context_recall_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    total_recall = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        results = row['Information']
        reply = row['Ground truth answer']
        
        # NOTE: Các em có thể thay đổi messages_judged nếu muốn 
        for result in results:
            messages_judged = [
                {
                    "role": "system",
                    "content": """Bạn là một trợ lý AI chuyên đánh giá độ chính xác của các câu trả lời dựa trên ngữ cảnh được cung cấp. Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trả lời đã được chuyên gia trả lời cho câu hỏi (đây là câu trả lời chính xác). Nhiệm vụ của bạn là đánh giá ngữ cảnh dựa trên câu hỏi và câu trả lời. Nếu ngữ cảnh và câu hỏi cung cấp đủ thông tin hoặc chỉ cần một phần thông tin để trả lời câu hỏi, hãy đánh giá ngữ cảnh là 1. Nếu không, hãy đánh giá là 0. Hãy đọc thật kĩ ngữ cảnh, chỉ cần ngữ cảnh có một phần thông tin để trả lời cho một phần của câu hỏi thì cũng đánh giá là 1. Nếu ngữ cảnh không liên quan gì đến câu hỏi, hãy đánh giá là 0. LƯU Ý: Chỉ trả lời 1 hoặc 0, không giải thích gì thêm."""
                }
            ]
            messages_judged.append({
                "role": "user",
                "content": f"Ngữ cảnh: {result}\n\nCâu hỏi: {query}\n\nCâu trả lời: {reply}"
            })

            judged_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_judged
            )
            judged_reply = judged_response.choices[0].message.content.strip()
            if judged_reply == "1":
                hits += 1
        recall = hits / k if k > 0 else 0
        total_recall += recall
    return total_recall / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Entities Recall@k (LLM Judged)
def context_entities_recall_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    total_recall = 0
    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        results = row['Information']
        reply = row['Ground truth answer']
        # Trích xuất các thực thể từ Ground truth answer bằng LLM
        # NOTE: Các em có thể thay đổi messages_entities nếu muốn
        messages_entities = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Yêu cầu bắt buộc: Chỉ trả lời dưới dạng một danh sách các thực thể, không giải thích gì thêm.
                Yêu cầu băt buộc: Các thực thể phải là các từ hoặc cụm từ có ý nghĩa, không bao gồm các từ dừng (stop words) như "là", "và", "của", v.v.
                Yêu cầu bắt buộc: Output phải là một list, ví dụ: ["thực thể 1", "thực thể 2", "thực thể 3"], và chuỗi đó phải eval được 
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            }
        ]
        # NOTE: Các em có thể thay đổi content nếu muốn
        # Gọi OpenAI API để trích xuất các thực thể
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        entities_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_entities
        )
        entities = entities_response.choices[0].message.content.strip().split("\n")
        entities = entities[0] # "["ngành khác", "CLB", "CNTT", "mảng]" -> ["ngành khác", "CLB", "CNTT", "mảng"]
        entities = eval(entities) if entities else []  # Chuyển đổi chuỗi thành danh sách
        tmp = len(entities)
        for result in results:
            context = result
            for entity in entities:
                if entity.strip() in context:
                    hits += 1
                    entities.remove(entity.strip())
        total_recall += hits / tmp if tmp > 0 else 0
    return total_recall / len(df_train) if len(df_train) > 0 else 0



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


# Các hàm đánh giá LLM Answer

# Hàm String Presence

def string_presence_k(file_clb_proptit, df_train, embedding, vector_db,  k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    # df_train = pd.read_excel(file_train)

    total_presence = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        results = row['Information']
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result for result in results])
        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            ).choices[0].message.content.strip()
        # Trích xuất các thực thể từ câu trả lời bằng LLM
        # NOTE: Các em có thể thay đổi message_entities nếu muốn
        messages_entities = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            }
        ]
        # Thay đổi content nếu muốn
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        # Gọi  API để trích xuất các thực thể
        entities = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_entities
        ).choices[0].message.content.strip().split("\n")
        entities = entities[0] # "["ngành khác", "CLB", "CNTT", "mảng]" -> ["ngành khác", "CLB", "CNTT", "mảng"]
        entities = eval(entities) if entities else []  # Chuyển đổi chuỗi thành danh sách
        for entity in entities:
            if entity.strip() in response:
                hits += 1
                # print(f"Entity '{entity.strip()}' found in response.")
        hits /= len(entities) if len(entities) > 0 else 0
        total_presence += hits
    return total_presence / len(df_train) if len(df_train) > 0 else 0


 

# Hàm Rouge-L

from rouge import Rouge
def rouge_l_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)

    rouge = Rouge()
    total_rouge_l = 0

    for index, row in df_train.iterrows():
        query = row['Query']

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = row['Information']
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result for result in results])

        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi API để lấy câu trả lời
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        ).choices[0].message.content.strip()
        scores = rouge.get_scores(response, reply)
        rouge_l = scores[0]['rouge-l']['f']
        total_rouge_l += rouge_l
    return total_rouge_l / len(df_train) if len(df_train) > 0 else 0

# Hàm BLEU-4
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu_4_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)

    total_bleu_4 = 0
    smoothing_function = SmoothingFunction().method1

    for index, row in df_train.iterrows():
        query = row['Query']
        results = row['Information']
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result for result in results])

        # Sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        response = response.choices[0].message.content.strip()
        reference = reply.split()
        candidate = response.split()
        bleu_4 = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
        total_bleu_4 += bleu_4
    return total_bleu_4 / len(df_train) if len(df_train) > 0 else 0


# Hàm Groundedness (LLM Answer - Hallucination Detection)\

def groundedness_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)

    total_groundedness = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        results = row['Information']
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result for result in results])

        # Thêm context vào messages, sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        response = response.choices[0].message.content.strip()
      
        # Tách response thành các câu
        sentences = response.split('. ')
        for sentence in sentences:
            # Tạo một prompt để kiểm tra tính groundedness của câu
            # NOTE: Các em có thể sửa đổi prompt này nếu muốn
            messages_groundedness = [
                {
                    "role": "system",
                    "content": """Bạn là một chuyên gia đánh giá Groundedness trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    supported: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    unsupported: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    contradictory: Nội dung câu trái ngược hoặc mâu thuẫn với ngữ cảnh.
                    no_rad: Câu không yêu cầu kiểm tra thực tế (ví dụ: câu chào hỏi, ý kiến cá nhân, câu hỏi tu từ, disclaimers).
                    Hãy trả lời bằng một trong các nhãn trên, không giải thích gì thêm. Chỉ trả lời một từ duy nhất là nhãn đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    supported"""
                }
            ]
            # Sửa content nếu muốn
            messages_groundedness.append({
                "role": "user",
                "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
            })
            # Gọi  API để đánh giá groundedness
            groundedness_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_groundedness
            )
            groundedness_reply = groundedness_response.choices[0].message.content.strip()
            # print("Câu:", sentence.strip())
            # print("Groundedness reply:", groundedness_reply)

            if groundedness_reply == "supported":
                hits += 1
                cnt += 1
            elif groundedness_reply == "unsupported" or groundedness_reply == "contradictory":
                cnt += 1
        total_groundedness += hits / cnt if cnt > 0 else 0
    return total_groundedness / len(df_train) if len(df_train) > 0 else 0 

# Hàm Response Relevancy (LLM Answer - Measures relevance)


def generate_related_questions(response):
    # Sửa systemp prompt nếu muốn
    messages_related = [
        {
            "role": "system",
            "content": """Bạn là một trợ lý AI chuyên tạo ra các câu hỏi liên quan từ một câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là tạo ra các câu hỏi liên quan đến câu trả lời đó. Hãy tạo ra ít nhất 5 câu hỏi liên quan, mỗi câu hỏi nên ngắn gọn và rõ ràng. Trả lời dưới dạng list các câu hỏi như ở ví dụ dưới. LƯU Ý: Trả lời dưới dạng ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3", ...], bao gồm cả dấu ngoặc vuông.
            Yêu cầu: Trong các câu hỏi không được chứa các kí tự đặc biệt làm cho câu hỏi không dùng hàm eval để trích xuất ra các thực thể được.
            Ví dụ:
            Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
            Output của bạn: "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
            Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
            Output của bạn: "["Ngành nào có thể tham gia CLB?", "CLB phù hợp với những ai?", "Trở ngại lớn nhất khi tham gia CLB là gì?"]"""
        }
    ]
    # Sửa content nếu muốn
    messages_related.append({
        "role": "user",
        "content": f"Câu trả lời: {response}"
    })
    # Gọi  API để tạo ra các câu hỏi liên quan
    related_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_related
    )
    related_questions = related_response.choices[0].message.content.strip()
    return related_questions

def response_relevancy_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)

    total_relevancy = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        user_embedding = embedding.encode(query)
        results = row['Information']
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.
    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result for result in results])

        # Sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        response = response.choices[0].message.content.strip()

        # Dùng câu trả lời của LLM để sinh ra các câu hỏi liên quan
        related_questions = generate_related_questions(response) # "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
        related_questions = eval(related_questions) if related_questions else []  # Chuyển đổi chuỗi thành danh sách
        for question in related_questions:
            question_embedding = embedding.encode(question)
            # Tính score relevancy giữa câu hỏi và query
            score = similarity(user_embedding, question_embedding)
            hits += score
        total_relevancy += hits / len(related_questions) if len(related_questions) > 0 else 0
    return total_relevancy / len(df_train) if len(df_train) > 0 else 0


# Hàm Noise Sensitivity (LLM Answer - Robustness to Hallucination)

def noise_sensitivity_k(file_clb_proptit, df_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)

    total_sensitivity = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        results = row['Information']
        reply = row['Ground truth answer']
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([result for result in results])
        # Thêm context vào messages, sửa content nếu muốn
        messages.append({
            "role": "user",
            "content": context + "\n\nCâu hỏi: " + query
        })
        # Gọi  API để lấy câu trả lời
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        response = response.choices[0].message.content.strip()

        sentences = response.split('. ')
        for sentence in sentences:
            # Sửa prompt nếu muốn
            messages_sensitivity = [
                {
                    "role": "system",
                    "content": """Bạn là một chuyên gia đánh giá độ nhạy cảm của câu trả lời trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    1: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    0: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    1
                    Question: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011. Bạn có biết ngày cụ thể không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT là CLB thuộc PTIT.
                    0"""
                }
            ]
            # Sửa prompt nếu muốn
            messages_sensitivity.append({
                "role": "user",
                "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
            })
            # Gọi  API để đánh giá độ nhạy cảm
            sensitivity_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_sensitivity
            )
            sensitivity_reply = sensitivity_response.choices[0].message.content.strip()
            if sensitivity_reply == "0":
                hits += 1
        total_sensitivity += hits / len(sentences) if len(sentences) > 0 else 0
    return total_sensitivity / len(df_train) if len(df_train) > 0 else 0


# Hàm để tính toán toàn bộ metrics trong module LLM Answer

def calculate_metrics_llm_answer(file_clb_proptit, file_train, embedding, vector_db, train):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "string_presence@k": [],
        "rouge_l@k": [],
        "bleu_4@k": [],
        "groundedness@k": [],
        "response_relevancy@k": [],
        "noise_sensitivity@k": []
    }
    print('CACULATING METRICS LLM ANSWER...')
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        print(f"Calculating metrics for k={k}...")
        metrics["K"].append(k)
        print("Creating embeddings and retrieving documents...")
        df_train=embedding_file(file_train, embedding, vector_db, k)
        print("Calculating string_presence@k...")
        metrics["string_presence@k"].append(round(string_presence_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print("Calculating rouge_l@k...")
        metrics["rouge_l@k"].append(round(rouge_l_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print("Calculating bleu_4@k...")
        metrics["bleu_4@k"].append(round(bleu_4_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print("Calculating groundedness@k...")
        metrics["groundedness@k"].append(round(groundedness_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print("Calculating response_relevancy@k...")
        metrics["response_relevancy@k"].append(round(response_relevancy_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print("Calculating noise_sensitivity@k...")
        metrics["noise_sensitivity@k"].append(round(noise_sensitivity_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print('-'*20)
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
    return metrics_df

def calculate_all_metrics(file_clb_proptit, file_train, embedding, vector_db, train):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    k_values = [7]
    retrieval_metrics = {
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
    
    llm_metrics = {
        "K": [],
        "string_presence@k": [],
        "rouge_l@k": [],
        "bleu_4@k": [],
        "groundedness@k": [],
        "response_relevancy@k": [],
        "noise_sensitivity@k": []
    }
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        print(f"Calculating metrics for k={k}...", end = ' ')
        print("Creating embeddings and retrieving documents...")
        df_train=embedding_file(file_train, embedding, vector_db, k)
    
        # print(f'Calculating retrieval metrics for k={k}...')
        # retrieval_metrics["K"].append(k)
        # print("Creating embeddings and retrieving documents...")
        # print("Calculating hit@k...", end = ' ')
        # retrieval_metrics["hit@k"].append(round(hit_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["hit@k"][-1])
        # print("Calculating recall@k...", end = ' ')
        # retrieval_metrics["recall@k"].append(round(recall_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["recall@k"][-1])
        # print("Calculating precision@k...", end = ' ')
        # retrieval_metrics["precision@k"].append(round(precision_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["precision@k"][-1])
        # print("Calculating f1@k...", end = ' ')
        # retrieval_metrics["f1@k"].append(round(f1_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["f1@k"][-1])
        # print("Calculating map@k...", end = ' ')
        # retrieval_metrics["map@k"].append(round(map_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["map@k"][-1])
        # print("Calculating mrr@k...", end = ' ')
        # retrieval_metrics["mrr@k"].append(round(mrr_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["mrr@k"][-1])
        # print("Calculating ndcg@k...", end = ' ')
        # retrieval_metrics["ndcg@k"].append(round(ndcg_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["ndcg@k"][-1])
        # print("Calculating context_precision@k...", end = ' ')
        # retrieval_metrics["context_precision@k"].append(round(context_precision_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["context_precision@k"][-1])
        # print("Calculating context_recall@k...", end = ' ')
        # retrieval_metrics["context_recall@k"].append(round(context_recall_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        # print(retrieval_metrics["context_recall@k"][-1])
        print("Calculating context_entities_recall@k...", end = ' ')
        retrieval_metrics["context_entities_recall@k"].append(round(context_entities_recall_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(retrieval_metrics["context_entities_recall@k"][-1])
        
        print('-'*20)
        print(f'Calculating LLM answer metrics for k={k}...')
        llm_metrics["K"].append(k)
        print("Calculating string_presence@k...", end = ' ')
        llm_metrics["string_presence@k"].append(round(string_presence_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(llm_metrics["string_presence@k"][-1])
        print("Calculating rouge_l@k...", end = ' ')
        llm_metrics["rouge_l@k"].append(round(rouge_l_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(llm_metrics["rouge_l@k"][-1])
        print("Calculating bleu_4@k...", end = ' ')
        llm_metrics["bleu_4@k"].append(round(bleu_4_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(llm_metrics["bleu_4@k"][-1])
        print("Calculating groundedness@k...", end = ' ')
        llm_metrics["groundedness@k"].append(round(groundedness_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(llm_metrics["groundedness@k"][-1])
        print("Calculating response_relevancy@k...", end = ' ')
        llm_metrics["response_relevancy@k"].append(round(response_relevancy_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(llm_metrics["response_relevancy@k"][-1])
        print("Calculating noise_sensitivity@k...", end = ' ')
        llm_metrics["noise_sensitivity@k"].append(round(noise_sensitivity_k(file_clb_proptit, df_train, embedding, vector_db, k), 2))
        print(llm_metrics["noise_sensitivity@k"][-1])
        print('-'*20)
    # Chuyển đổi metrics thành DataFrame
    
    retrieval_metrics_df = pd.DataFrame(retrieval_metrics)
    llm_metrics_df = pd.DataFrame(llm_metrics)
    # Lưu DataFrame vào file csv
    # if train:
    #     retrieval_metrics_df.to_csv("metrics_retrieval_train.csv", index=False)
    #     llm_metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
    # else:
    #     retrieval_metrics_df.to_csv("metrics_retrieval_test.csv", index=False)
    #     llm_metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
    return retrieval_metrics_df, llm_metrics_df