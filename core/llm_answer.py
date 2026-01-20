import os
import logging  # Gunakan logging instead of print
from groq import Groq
from typing import Optional
from dotenv import load_dotenv

# --- Konfigurasi Logging ---
# Siapkan logger sekali di awal file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMAnswerGenerator:
    """
    Menggunakan Groq untuk menghasilkan jawaban berdasarkan 
    query dan konteks yang diberikan (RAG).
    """
    def __init__(self, model_name: Optional[str] = None):
        """
        Inisialisasi generator jawaban menggunakan Groq.
        
        Args:
            model_name (Optional[str]): Nama model Groq yang akan digunakan.
                Jika None, akan menggunakan model default dari konfigurasi.
        """
        load_dotenv()
        self.api_key = os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            logger.error("API Key Groq tidak ditemukan. Pastikan file .env berisi variabel GROQ_API_KEY=...")
            raise ValueError("API Key Groq tidak ditemukan.")

        # Tentukan model default jika tidak ada yang dipilih
        # Ini membuatnya lebih mudah untuk mengganti model default di satu tempat
        default_model = "llama-3.3-70b-versatile" # Contoh model Llama 3 dari Groq
        self.model_name = model_name or default_model

        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"LLMAnswerGenerator berhasil diinisialisasi dengan model: {self.model_name}")
        except Exception as e:
            logger.error(f"Gagal menginisialisasi LLMAnswerGenerator: {e}")
            self.client = None

    def generate_answer(self, query: str, context: str) -> str:
        """
        Menghasilkan jawaban berdasarkan query dan konteks menggunakan Groq.

        Args:
            query (str): Pertanyaan dari pengguna.
            context (str): Konteks atau dokumen yang relevan untuk menjawab query.

        Returns:
            str: Jawaban yang dihasilkan oleh model atau pesan error.
        """
        if not self.client:
            logger.error("Model LLM Groq tidak tersedia karena klien gagal diinisialisasi.")
            return "Maaf, layanan AI sedang bermasalah. Silakan coba lagi nanti."

        # Validasi input
        if not query or not query.strip():
            logger.warning("Query kosong atau hanya berisi whitespace.")
            return "Maaf, pertanyaan tidak boleh kosong."
        
        if not context or not context.strip():
            logger.warning("Context kosong atau hanya berisi whitespace.")
            return "Maaf, tidak ada konteks yang tersedia untuk menjawab pertanyaan Anda."

        # Prompt yang lebih terstruktur
        system_message = "Anda adalah asisten cerdas yang menjawab berdasarkan dokumen yang diberikan."
        user_prompt = f"""
        Kamu adalah asisten AI yang menjawab pertanyaan berdasarkan **hanya** konteks yang diberikan.
        Jawablah dengan ringkas, jelas, dan hanya menggunakan informasi dari konteks.
        Susun jawaban dalam bentuk daftar dengan bullet point (misalnya menggunakan * atau -).
        Jika jawaban tidak ditemukan dalam konteks, katakan dengan sopan "Maaf, saya tidak menemukan informasi tersebut dalam dokumen."

        Konteks:
        ---
        {context}
        ---

        Pertanyaan: {query}

        Jawaban:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Berhasil menghasilkan jawaban untuk query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Terjadi kesalahan saat menghasilkan jawaban via Groq: {e}")
            return "Maaf, terjadi kesalahan internal saat memproses permintaan Anda. Silakan coba lagi."