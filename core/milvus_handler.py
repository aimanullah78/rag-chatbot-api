import sys
from pymilvus import MilvusClient

# --- Buat kelas "palsu" untuk meniru format hasil pencarian lama ---
class SearchResultWrapper:
    """Sebuah wrapper untuk membuat hasil pencarian pymilvus kompatibel dengan script lama."""
    def __init__(self, hit_data):
        self.id = hit_data.get('id')
        self.distance = hit_data.get('distance')
        self.entity = hit_data # 'entity' adalah dictionary hasil itu sendiri

class MilvusHandler:
    def __init__(self, config):
        self.config = config
        self.collection_name = config.get("collection_name")
        self.vector_field = config.get("vector_field_name")
        self.primary_field = config.get("primary_field_name")
        self.text_field = config.get("text_field_name")
        self.metadata_field = config.get("metadata_field_name") # Ambil nama field metadata

        try:
            self.client = MilvusClient(
                uri=config.get("uri"),
                token=config.get("token")
            )
            self.client.load_collection(self.collection_name)
            print(f"[MilvusHandler] Berhasil terhubung ke collection '{self.collection_name}' di Zilliz Cloud.", file=sys.stderr)
        except Exception as e:
            print(f"[MilvusHandler] Gagal terhubung ke Zilliz Cloud: {e}", file=sys.stderr)
            self.client = None
            raise

    def search(self, query_vector, top_k=10):
        if not self.client:
            print("[MilvusHandler] Koneksi tidak ada.", file=sys.stderr)
            return []

        try:
            # Tentukan field mana yang akan diambil
            output_fields = [self.text_field, self.primary_field]
            if self.metadata_field:
                output_fields.append(self.metadata_field)

            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field=self.vector_field,
                limit=top_k,
                output_fields=output_fields
            )

            # Format hasil menjadi list dari SearchResultWrapper
            wrapped_results = []
            for hit in results[0]:
                # Data utama ada di hit['entity']
                entity_data = hit.get('entity', {})
                
                # Buat dictionary yang akan menjadi 'entity' di wrapper
                # Ini memastikan 'text' dan 'metadata' ada di tingkat atas
                final_entity_data = {
                    self.primary_field: entity_data.get(self.primary_field),
                    self.text_field: entity_data.get(self.text_field),
                    self.metadata_field: entity_data.get(self.metadata_field, {})
                }
                
                # Buat objek wrapper
                wrapped_hit = SearchResultWrapper({
                    'id': entity_data.get(self.primary_field),
                    'distance': hit.get('distance'),
                    'entity': final_entity_data
                })
                wrapped_results.append(wrapped_hit)
            
            return wrapped_results

        except Exception as e:
            print(f"[MilvusHandler] Error saat pencarian: {e}", file=sys.stderr)
            return []

    def close(self):
        print("[MilvusHandler] Koneksi ditutup.", file=sys.stderr)