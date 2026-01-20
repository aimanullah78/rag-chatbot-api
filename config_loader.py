import os
import json

def load_config():
    """
    Membaca konfigurasi dari Environment Variables terlebih dahulu.
    Jika tidak ada, fallback ke file config.json (untuk pengembangan lokal).
    """
    # Prioritas 1: Coba dari Environment Variables (untuk produksi/cloud)
    milvus_uri = os.environ.get('MILVUS_URI')
    milvus_token = os.environ.get('MILVUS_TOKEN')
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if milvus_uri and milvus_token and groq_api_key:
        return {
            'milvus': {
                'uri': milvus_uri,
                'token': milvus_token
            },
            'llm': {
                'provider': 'groq',
                'api_key': groq_api_key
            }
        }

    # Prioritas 2: Fallback ke file config.json (untuk pengembangan lokal)
    try:
        with open('config.json', 'r') as f:
            print("Warning: Environment variables not found, falling back to config.json for local development.")
            return json.load(f)
    except FileNotFoundError:
        print("Error: Neither environment variables nor config.json found.")
        return {} # atau sys.exit(1)