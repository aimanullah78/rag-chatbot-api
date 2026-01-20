# api_server.py (SUDAH DITAMBAHKAN ENDPOINT GAMBAR)

from flask import Flask, request, jsonify, send_from_directory  # <--- TAMBAHKAN send_from_directory
from flask_cors import CORS
from chatbot_service import ChatbotService
import os

# --- INISIALISASI UTAMA ---
print("Starting server and initializing ChatbotService...")
try:
    chatbot_service = ChatbotService()
    print("ChatbotService is ready.")
except Exception as e:
    print(f"Failed to initialize ChatbotService: {e}")
    chatbot_service = None

# Buat aplikasi Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://milvus_web.localhost", "http://localhost:5000", "http://127.0.0.1:5000", "http://192.168.100.66:5000"]}})

# --- ENDPOINT API ---

@app.route('/', methods=['GET'])
def index():
    """Endpoint untuk mengecek apakah server berjalan."""
    return jsonify({
        "message": "RAG Chatbot API is running!",
        "status": "healthy" if chatbot_service else "unhealthy"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint utama untuk menerima pertanyaan dari pengguna."""
    if not chatbot_service:
        return jsonify({"error": "Service is not initialized. Check server logs."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. JSON body is missing."}), 400

    query = data.get('query')
    if not query or not isinstance(query, str):
        return jsonify({"error": "Missing 'query' field in request or it's not a string."}), 400

    history = data.get('history', [])
    if not isinstance(history, list):
        return jsonify({"error": "'history' field must be a list."}), 400

    print(f"Received query: {query}")

    try:
        response = chatbot_service.get_response(query, history)
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Endpoint untuk membersihkan riwayat percakapan di sisi server."""
    if not chatbot_service:
        return jsonify({"error": "Service is not initialized."}), 503
    
    chatbot_service.clear_history()
    return jsonify({"status": "Conversation history cleared on the server."})


# <--- TAMBAHKAN ENDPOINT BARU INI --->
@app.route('/source_image/<path:filename>')
def serve_source_image(filename):
    """
    Endpoint untuk melayani file gambar dari folder 'output'.
    Contoh URL: /source_image/Manual_Akuntansi/images/p13_full.png
    """
    # 'output' adalah nama folder di mana gambar disimpan
    # 'filename' adalah path relatif di dalam folder tersebut
    return send_from_directory('output', filename)


# --- MENJALANKAN SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)