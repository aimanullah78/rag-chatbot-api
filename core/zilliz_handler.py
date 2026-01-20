# zilliz_handler.py
from pymilvus import connections, Collection

# Hapus import SentenceTransformer dan CrossEncoder dari sini
# from sentence_transformers import SentenceTransformer, CrossEncoder

class ZillizHandler:
    def __init__(self, config, embedding_model, reranker_model):
        """
        Inisialisasi handler dengan menerima model yang sudah ada.
        """
        self.collection_name = config['collection_name']
        self.uri = config['uri']
        self.token = config['token']
        
        # --- PERUBAHAN: Gunakan model yang sudah ada, bukan buat baru ---
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        
        self.collection = None
        self._connect()

    def _connect(self):
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token
            )
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print("✅ Berhasil terhubung ke Zilliz Cloud.")
        except Exception as e:
            print(f"❌ Gagal terhubung ke Zilliz Cloud: {e}")
            raise e

    def search(self, query: str, top_k: int = 10):
        """Melakukan pencarian vektor dan reranking."""
        # 1. Encode query menjadi vektor (menggunakan model yang sudah ada)
        query_vector = self.embedding_model.encode(query).tolist()

        # 2. Lakukan pencarian di Zilliz Cloud
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "chunk_id", "document_source", "jenis_dokumen", "judul_bab", "bab", "source_file", "halaman_awal", "halaman_akhir"]
        )

        if not results or not results[0]:
            return []

        # 3. Format hasil ke dalam dictionary yang konsisten
        hits = []
        for hit in results[0]:
            entity = hit.entity
            metadata = {
                "source_file": entity.get("source_file"),
                "page": entity.get("halaman_awal"),
                "judul_bab": entity.get("judul_bab"),
                "bab": entity.get("bab"),
                "jenis_dokumen": entity.get("jenis_dokumen")
            }
            hits.append({
                'id': hit.id,
                'distance': hit.distance,
                'text': entity.get("text"),
                'metadata': metadata
            })
        
        # 4. Rerank hasil (menggunakan model yang sudah ada)
        if not hits: return []
        passage_pairs = [[query, hit.get('text', '')[:1500]] for hit in hits]
        try:
            scores = self.reranker_model.predict(passage_pairs)
            for i, hit in enumerate(hits): hit['rerank_score'] = float(scores[i])
            hits.sort(key=lambda x: x['rerank_score'], reverse=True)
        except Exception as e:
            print(f"[WARNING] Rerank error: {e}")
        
        return hits