from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name: str = "namdp-ptit/ViRanker", use_fp16: bool = True, normalize: bool = True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        self.normalize = normalize

    def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
        # Tạo cặp [query, passage] cho mỗi passage
        query_passage_pairs = [[query, passage] for passage in passages]

        # Tính điểm từ reranker model
        scores = self.reranker.compute_score(query_passage_pairs, normalize=self.normalize)

        # Sắp xếp passage theo điểm số giảm dần
        ranked_data = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        ranked_scores, ranked_passages = zip(*ranked_data)

        # Đảm bảo đầu ra là list chuẩn
        return list(ranked_scores), list(ranked_passages)