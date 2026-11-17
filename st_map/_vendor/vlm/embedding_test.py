from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
class EmbeddingText():
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cuda")
        self.target_feat = None

    @property
    def embedding_model(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=self.model_name,
                                     model_kwargs={"device": "cuda"},
                                     encode_kwargs={"normalize_embeddings": True})
    def retrieval(self,query_txt,documents):
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.create_documents(
            list(documents.values()),
            metadatas=[{"original_index": index} for index in documents.keys()]
        )
        vector_db = FAISS.from_documents(texts, self.embedding_model)
        results = vector_db.similarity_search_with_relevance_scores(query_txt, k=5)
        return results
    def target_query(self,query_txt,classes):
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.create_documents(
            classes,
            metadatas=[{"original_index": index} for index in range(len(classes))]
        )
        vector_db = FAISS.from_documents(texts, self.embedding_model)
        results = vector_db.similarity_search_with_relevance_scores(query_txt, k=5)
        return results
    def query(self,text1,text2):
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity
    def query_multiple(self,texts):
        embedding1 = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        similarity = F.cosine_similarity(embedding1, self.target_feat , dim=1)
        value= torch.max(similarity)
        all_max_indices = torch.where(similarity == value)[0]
        return value.item(),len(all_max_indices)
    def compute_target_feat(self,target_object):
        embedding2 = self.model.encode(target_object, convert_to_tensor=True, show_progress_bar=False)
        self.target_feat = embedding2.unsqueeze(0)
# print(f"Semantic similarity between Trash bin Can and GarbageCan: {similarity.item():.4f}")