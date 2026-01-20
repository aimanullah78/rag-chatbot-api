# chatbot_service.py (VERSI LENGKAP DAN BENAR)

import os
import sys
import re
import uuid
import json
import ast

# --- IMPORTS YANG SUDAH DISESUAIKAN ---
try:
    # Import handler baru untuk Zilliz Cloud
    from core.zilliz_handler import ZillizHandler
    from core.llm_answer import LLMAnswerGenerator
    # Fungsi load_config diasumsikan bisa membaca config.json
    from config_loader import load_config 
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    # Di sini kita tidak bisa pakai messagebox, jadi kita print error dan hentikan
    print(f"Gagal mengimpor modul: {e}\nPastikan library yang dibutuhkan terinstall.")
    sys.exit(1)

# --- KONFIGURASI ---
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
BASE_OUTPUT_DIR = "output" 
MAX_HISTORY_TURNS = 5
ENUMERATION_SEARCH_TOP_K = 50
BASE_API_URL = "http://192.168.100.66:5000"

# --- DAFTAR KATA KUNCI UNTUK PERCAKAPAN UMUM ---
CONVERSATIONAL_KEYWORDS = [
    "hai", "halo", "apa kabar", "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "terima kasih", "makasih", "thanks", "siapa kamu", "kamu siapa", "apa ini", "bantuan", "help",
    "apa yang bisa kamu lakukan", "sampai jumpa", "dadah"
]

class ChatbotService:
    def __init__(self):
        """
        Inisialisasi semua komponen yang diperlukan.
        Model-model yang berat akan dimuat sekali di sini.
        """
        print("Initializing ChatbotService...")
        
        # Inisialisasi komponen utama
        self.milvus = None
        self.llm_generator = None
        self.embedding_model = None
        self.reranker_model = None
        self.conversation_history = []

        # Di sinilah kita akan memindahkan logika dari 'setup_components'
        self._load_models_and_handlers()
        
        print("ChatbotService initialization complete.")

    def _load_models_and_handlers(self):
        """
        Memuat model-model dan handler dari Zilliz.
        """
        try:
            print("Loading configuration from config.json...")
            config = load_config()
            milvus_config = config['milvus']
            
            print(f"Loading AI Models...")
            # Model dimuat di sini
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
            print("AI Models loaded.")
            
            # --- PERUBAHAN UTAMA: Pass config DAN model yang sudah ada ---
            self.milvus = ZillizHandler(milvus_config, self.embedding_model, self.reranker_model)
            
            self.llm_generator = LLMAnswerGenerator()
            
            # Tidak perlu pesan welcome atau GUI di sini
            print("All components loaded successfully.")

        except Exception as e:
            print(f"Error during service initialization: {e}")
            # Hentikan eksekusi jika komponen gagal dimuat
            sys.exit(1)

    # --- METODE UTAMA UNTUK DIPANGGIL OLEH API ---
    def get_response(self, query: str, history: list):
        """
        Metode utama untuk memproses query dari API.
        Menerima query dan riwayat, lalu mengembalikan jawaban, sumber, dan saran.
        """
        # Gunakan riwayat dari request, bukan riwayat internal
        self.conversation_history = history
        
        if not self.milvus:
            return {"error": "Service is not fully initialized yet."}

        is_conversational = self._is_conversational_query(query)
        if is_conversational:
            response = self._generate_conversational_response(query)
            self._add_to_history("user", query)
            self._add_to_history("bot", response)
            return {
                "answer": response,
                "sources": [],
                "suggestions": [],
                "updated_history": self.conversation_history
            }

        is_comparison = self._is_comparison_query(query)
        query_type = self._classify_query_type(query)

        result = {}
        if is_comparison:
            result = self.process_comparison_query(query)
        elif query_type == "ENUMERATION":
            result = self._process_enumeration_query(query)
        else:
            result = self.process_standard_query(query)

        # Tambahkan ke riwayat
        self._add_to_history("user", query)
        self._add_to_history("bot", result.get("answer", "Maaf, saya tidak bisa menjawab."))

        # Generate suggestions based on the final context
        context_for_suggestion = "\n\n".join([hit.get('text', '') for hit in result.get("sources", [])])
        suggestions = self._generate_proactive_suggestions(query, context_for_suggestion)

        return {
            "answer": result.get("answer", "Maaf, terjadi kesalahan internal."),
            "sources": self._format_sources_for_api(result.get("sources", [])), # <--- MEMANGGIL FUNGSI YANG AKAN KITA BUAT
            "suggestions": suggestions,
            "updated_history": self.conversation_history
        }

    def clear_history(self):
        """Membersihkan riwayat percakapan."""
        self.conversation_history = []
        return {"status": "Conversation history cleared."}

    # --- METODE PEMBANTU (TIDAK BERUBAH BANYAK) ---
    def _add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def _format_history_for_prompt(self) -> str:
        if not self.conversation_history: return ""
        recent_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]
        formatted_lines = []
        for item in recent_history:
            role = "Pengguna" if item['role'] == 'user' else "Asisten"
            formatted_lines.append(f"{role}: {item['content']}")
        return "\n".join(formatted_lines)

    def _classify_query_type(self, query: str) -> str:
        enumeration_keywords = ["siapa saja", "apa saja", "daftar", "semua", "seluruh", "kumpulan", "berikut", "sebutkan"]
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in enumeration_keywords):
            return "ENUMERATION"
        return "FACT"

    def _build_aggregation_prompt(self, query: str, context: str) -> str:
        return f"""Tugas Anda adalah menjadi agregator informasi yang teliti. Dari kumpulan teks dokumen di bawah ini, ekstrak SEMUA item yang relevan dengan permintaan pengguna.
Permintaan Pengguna: '{query}'
Konteks Dokumen:
{context}
INSTRUKSI:
1. Baca SEMUA konteks dengan sangat teliti. Jangan lewatkan informasi.
2. Temukan dan kumpulkan SEMUA entitas (nama, topik, item, dll.) yang cocok dengan permintaan.
3. Jawaban harus berupa daftar yang lengkap dan komprehensif.
4. Jika ada duplikat, gabungkan menjadi satu.
5. Keluarkan jawaban HANYA dalam bentuk daftar (bullet points) tanpa teks pembuka atau penutup.
Contoh format jawaban:
- Item A
- Item B
- Item C
"""

    def _generate_fallback_suggestions(self, raw_text: str) -> list:
        questions = re.findall(r'[^.!?]*\?', raw_text)
        if questions:
            return [q.strip() for q in questions if q.strip()][:3]
        return ["Bisa jelaskan lebih detail tentang topik ini?", "Apa implikasi dari informasi ini?", "Apakah ada contoh kasus yang relevan?"]

    def _generate_proactive_suggestions(self, original_query: str, context: str) -> list:
        prompt = f"""Anda adalah asisten AI. Tugas Anda adalah membuat 3 (tiga) saran pertanyaan lanjutan yang relevan berdasarkan konteks yang diberikan.
        ATURAN PENTING:
        1. Jawaban HARUS berupa sebuah list Python yang valid.
        2. Setiap elemen dalam list adalah sebuah string yang merupakan pertanyaan.
        3. Jangan gunakan nomor atau format lain, hanya list Python.
        4. Pertanyaan harus spesifik dan bisa dijawab oleh dokumen.
        CONTOH FORMAT YANG BENAR:
        ['Apa topik utama dari dokumen X?', 'Kebijakan apa yang berubah di tahun Y?', 'Siapa yang bertanggung jawab atas Z?']
        ---
        Pertanyaan Pengguna: {original_query}
        ---
        Konteks Dokumen Relevan:
        {context[:2000]}
        ---
        Saran Pertanyaan Lanjutan (HANYA keluarkan list Python-nya saja, tanpa teks tambahan):
        """
        try:
            # Berikan query asli sebagai parameter pertama, meskipun tidak digunakan dalam prompt, untuk konsistensi
            suggestions_text = self.llm_generator.generate_answer(original_query, prompt).strip()
            if suggestions_text.startswith("```python"):
                suggestions_text = suggestions_text.replace("```python", "").replace("```", "").strip()
            suggestions = ast.literal_eval(suggestions_text)
            if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                return suggestions
        except (SyntaxError, ValueError) as e:
            print(f"[WARNING] Gagal parsing saran dari LLM: {e}. Respons mentah: {suggestions_text}")
            return self._generate_fallback_suggestions(suggestions_text)
        except Exception as e:
            print(f"[ERROR] Gagal menghasilkan saran: {e}")
            return []
        return []

    def _is_conversational_query(self, query: str) -> bool:
        query_lower = query.lower()
        for keyword in CONVERSATIONAL_KEYWORDS:
            if keyword in query_lower:
                return True
        return False

    def _generate_conversational_response(self, query: str) -> str:
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["hai", "halo"]):
            return "Halo! Ada yang bisa saya bantu terkait dokumen Anda hari ini?"
        if any(kw in query_lower for kw in ["apa kabar"]):
            return "Kabar saya baik, terima kasih! Saya siap membantu Anda mencari informasi dari dokumen."
        if any(kw in query_lower for kw in ["terima kasih", "makasih", "thanks"]):
            return "Sama-sama! Senang bisa membantu. Jika ada pertanyaan lain, jangan ragu untuk bertanya."
        if any(kw in query_lower for kw in ["siapa kamu", "kamu siapa"]):
            return "Saya adalah asisten AI cerdas yang dirancang untuk membantu Anda mencari, membandingkan, dan menganalisis informasi dari dokumen perusahaan."
        if any(kw in query_lower for kw in ["apa yang bisa kamu lakukan", "bantuan", "help"]):
            return ("Saya bisa membantu Anda dengan:\n"
                    "1. Menjawab pertanyaan berdasarkan dokumen.\n"
                    "2. Membandingkan dua dokumen atau topik.\n"
                    "3. Mengingat percakapan kita dalam sesi ini.\n\n"
                    "Coba tanyakan sesuatu!")
        if any(kw in query_lower for kw in ["sampai jumpa", "dadah"]):
            return "Sampai jumpa! Semoga harimu menyenangkan."
        return "Maaf, saya tidak yakin maksud Anda. Saya dirancang untuk membantu pertanyaan seputar dokumen. Coba ajukan pertanyaan atau katakan 'bantuan' untuk melihat apa yang bisa saya lakukan."

    def _is_comparison_query(self, query: str) -> bool:
        keywords = ["bandingkan", "perbedaan", "persamaan", "vs", "dibanding", "perubahan dari", "evolusi", "bedanya"]
        return any(keyword in query.lower() for keyword in keywords)

    def _extract_entities_for_comparison(self, query: str) -> list:
        query_lower = query.lower()
        separators = [" dan ", " vs ", " dengan ",", "]
        entities = []
        for sep in separators:
            if sep in query_lower:
                parts = query_lower.split(sep)
                if len(parts) >= 2:
                    part1 = parts[0].replace("bandingkan", "").strip()
                    part2 = parts[1].strip()
                    entities.extend([part1, part2])
                    break
        if not entities:
            match = re.search(r"dari (.+) ke (.+)", query_lower)
            if match:
                entities = [match.group(1), match.group(2)]
        return list(filter(None, list(set(entities))))

    # --- METODE PEMROSESAN (SUDAH DIMODIFIKASI UNTUK MENGEMBALIKAN DATA) ---
    def process_standard_query(self, query: str):
        print(f"Searching for: {query}")
        
        all_hits = self.milvus.search(query, top_k=30)

        if not all_hits:
            return {"answer": "Maaf, tidak ada dokumen ditemukan untuk pertanyaan tersebut.", "sources": []}

        reranked_hits = self._ai_rerank_results(query, all_hits)
        
        final_hits = []
        if reranked_hits:
            final_hits.append(reranked_hits[0])
            for hit in reranked_hits[1:5]:
                if hit.get('rerank_score', -99) > 0.01:
                    final_hits.append(hit)

        context_list = []
        for hit in final_hits:
            raw_text = hit.get('text', '')
            meta = hit.get('metadata', {})
            source_info = f"[Sumber: {meta.get('source_file')} Halaman {meta.get('page')}]"
            context_list.append(f"{source_info}\n{raw_text}")
        
        full_context = "\n\n".join(context_list)
        history_string = self._format_history_for_prompt()
        prompt = self._build_contextual_prompt(query, history_string, full_context)
        
        answer = self.llm_generator.generate_answer(query, prompt)
        return {"answer": answer, "sources": final_hits}

    def _build_contextual_prompt(self, query: str, history: str, context: str) -> str:
        if history:
            prompt = f"""Kamu adalah asisten AI yang membantu pengguna menjawab pertanyaan berdasarkan dokumen. Gunakan riwayat percakapan untuk memahami konteks dari pertanyaan terbaru pengguna.
---
RIWAYAT PERCAKAPAN SEBELUMNYA:
{history}
---
PERTANYAAN TERBARU PENGGUNA:
{query}
---
INFORMASI DARI DOKUMEN YANG RELEVAN:
{context}
---
Berdasarkan riwayat dan informasi dokumen di atas, jawablah pertanyaan terbaru pengguna secara jelas dan ringkas. Jika pertanyaan terkait dengan topik sebelumnya, jelaskan hubungannya."""
        else:
            prompt = f"Jawab pertanyaan berikut berdasarkan konteks yang diberikan.\n\nPertanyaan: {query}\n\nKonteks:\n{context}\n\nJawaban:"
        return prompt

    def _process_enumeration_query(self, query: str):
        print(f"Processing ENUMERATION query: {query}")
        
        all_hits = []
        seen_ids = set()

        query_vector = self.embedding_model.encode(query).tolist()
        vector_hits_raw = self.milvus.search(query_vector, top_k=ENUMERATION_SEARCH_TOP_K)
        
        if vector_hits_raw:
            for hit in vector_hits_raw:
                extracted_hit = {}
                if isinstance(hit, dict): extracted_hit = hit
                elif hasattr(hit, 'entity'):
                    try:
                        entity = hit.entity
                        if hasattr(entity, 'to_dict'): entity = entity.to_dict()
                        extracted_hit = {'id': hit.id, 'distance': hit.distance, 'text': entity.get('text'), 'metadata': entity.get('metadata')}
                    except: continue
                
                if not extracted_hit.get('text'): continue
                meta = extracted_hit.get('metadata', {})
                source = meta.get('source_file')
                if not source: continue
                uid = f"{source}_{meta.get('page')}"
                if uid not in seen_ids:
                    all_hits.append(extracted_hit)
                    seen_ids.add(uid)
        
        if not all_hits:
            return {"answer": "Maaf, tidak ada dokumen ditemukan untuk pertanyaan tersebut.", "sources": []}

        reranked_hits = self._ai_rerank_results(query, all_hits)
        context_list = []
        for hit in reranked_hits:
            raw_text = hit.get('text', '')
            meta = hit.get('metadata', {})
            source_info = f"[Sumber: {meta.get('source_file')} Halaman {meta.get('page')}]"
            context_list.append(f"{source_info}\n{raw_text}")
        
        full_context = "\n\n".join(context_list)
        prompt = self._build_aggregation_prompt(query, full_context)
        answer = self.llm_generator.generate_answer(query, prompt)
        return {"answer": answer, "sources": reranked_hits[:10]}

    def process_comparison_query(self, query: str):
        entities = self._extract_entities_for_comparison(query)
        if len(entities) < 2:
            return {"answer": "Maaf, saya tidak yakin apa yang ingin Anda bandingkan. Tolong sebutkan dua dokumen atau topik.", "sources": []}
        
        all_comparison_contexts = {}
        all_comparison_sources = {}

        for entity in entities:
            print(f"Searching for context of: {entity}")
            all_hits = self.milvus.search(entity, top_k=15)
            
            reranked_hits = self._ai_rerank_results(entity, all_hits)
            
            context_list = []
            for hit in reranked_hits[:3]:
                context_list.append(hit.get('text', ''))
            
            all_comparison_contexts[entity] = "\n\n".join(context_list)
            all_comparison_sources[entity] = reranked_hits[:3]

        history_string = self._format_history_for_prompt()
        prompt = self._build_comparison_prompt(query, history_string, all_comparison_contexts)
        answer = self.llm_generator.generate_answer("", prompt)
        
        # Gabungkan semua sumber untuk ditampilkan
        combined_sources = []
        for sources in all_comparison_sources.values():
            combined_sources.extend(sources)
        
        return {"answer": answer, "sources": combined_sources}

    def _build_comparison_prompt(self, original_query: str, history: str, contexts: dict) -> str:
        entity_keys = list(contexts.keys())
        context1 = contexts.get(entity_keys[0], 'Informasi tidak ditemukan.')
        context2 = contexts.get(entity_keys[1], 'Informasi tidak ditemukan.')
        prompt = f"""Kamu adalah asisten AI yang ahli dalam menganalisis dokumen. Gunakan riwayat percakapan untuk memahami konteks dari permintaan perbandingan ini.
---
RIWAYAT PERCAKAPAN SEBELUMNYA:
{history}
---
PERMINTAAN PERBANDINGAN PENGGUNA:
{original_query}
---
INFORMASI DARI SUMBER PERTAMA ({entity_keys[0]}):
{context1}
---
INFORMASI DARI SUMBER KEDUA ({entity_keys[1]}):
{context2}
---
Berdasarkan riwayat dan informasi di atas, buatlah analisis perbandingan yang terstruktur dengan jelas dalam Bahasa Indonesia:
1.  **Persamaan Utama:** Jelaskan titik-titik yang sama antara kedua sumber.
2.  **Perbedaan Kunci:** Jelaskan perbedaan signifikan secara point-by-point.
3.  **Analisis Perubahan/Tren:** Jelaskan implikasi atau tren dari perbedaan tersebut.
Jawaban harus ringkas, objektif, dan hanya berdasarkan informasi yang diberikan."""
        return prompt

    def _ai_rerank_results(self, query: str, hits: list):
        if not hits: return []
        valid_hits = [h for h in hits if isinstance(h, dict) and 'text' in h]
        if not valid_hits: return []
        passage_pairs = [[query, hit.get('text', '')[:1500]] for hit in valid_hits]
        try:
            scores = self.reranker_model.predict(passage_pairs)
            for i, hit in enumerate(valid_hits): hit['rerank_score'] = float(scores[i])
            valid_hits.sort(key=lambda x: x['rerank_score'], reverse=True)
        except Exception as e:
            print(f"[WARNING] Rerank error: {e}")
            return valid_hits 
        return valid_hits

    # <--- INI ADALAH FUNGSI YANG HILANG. PASTIKAN ADA DI DALAM KELAS --->
# Di dalam chatbot_service.py

    def _format_sources_for_api(self, sources: list):
        """Memformat data sumber agar lebih rapi untuk dikirim via API, termasuk URL gambar."""
        formatted_sources = []
        processed_pages = set()
        for source in sources:
            metadata = source.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown.pdf')
            page_num = metadata.get('page', '?')
            page_key = (source_file, page_num)
            if page_key in processed_pages: continue
            
            processed_pages.add(page_key)
            
            # --- LOGIKA UNTUK MENEMUKAN GAMBAR ---
            image_url = None
            folder_name = os.path.splitext(source_file)[0]
            images_dir = os.path.join(BASE_OUTPUT_DIR, folder_name, "images")
            
            try:
                # Coba nama file yang paling mungkin
                img_id = f"p{page_num}_full"
                img_path = os.path.join(images_dir, f"{img_id}.png")
                if os.path.exists(img_path):
                    # Jika ditemukan, buat URLnya MENGGUNAKAN BASE_API_URL
                    image_url = f"{BASE_API_URL}/source_image/{folder_name}/images/{img_id}.png"
                else:
                    # Fallback ke nama lain
                    guesses = [f"page_{page_num}.png", f"{page_num}.png", f"p{page_num}_img_0.png"]
                    for guess in guesses:
                        path_check = os.path.join(images_dir, guess)
                        if os.path.exists(path_check):
                            image_url = f"{BASE_API_URL}/source_image/{folder_name}/images/{guess}" # <--- JUGA DIUBAH DI SINI
                            break
            except Exception as e:
                print(f"[ERROR IMAGE] {e}")
                image_url = None
            
            # --- AKHIR LOGIKA GAMBAR ---

            formatted_sources.append({
                "source_file": source_file,
                "page": page_num,
                "relevance_score": source.get('rerank_score', 0.0),
                "text_snippet": source.get('text', '')[:200] + "...",
                "image_url": image_url
            })
        return formatted_sources