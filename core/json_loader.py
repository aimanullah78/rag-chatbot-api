import os
import json
import glob

class JSONCorpusLoader:
    """
    Memuat data dari file-file JSON di direktori output.
    Data ini digunakan terutama untuk keyword search.
    """
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir
        self.corpus = {}  # Akan menyimpan data dengan format: {(filename, page): [data_list]}

        if not os.path.exists(self.base_output_dir):
            print(f"[WARNING] JSONCorpusLoader: Directory {self.base_output_dir} not found. Keyword search will be disabled.")
            return

        print(f"[JSONCorpusLoader] Loading corpus from {self.base_output_dir}...")
        self._load_corpus()
        print(f"[JSONCorpusLoader] Finished loading. Total pages loaded: {len(self.corpus)}")

    def _load_corpus(self):
        # Cari semua file dengan pola nama 'folder_namafile/namafile_o_dt.json'
        search_pattern = os.path.join(self.base_output_dir, '*', '*_o_dt.json')
        json_files = glob.glob(search_pattern)

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Ekstrak nama folder dan nama file dari path
                full_path_parts = json_path.split(os.sep)
                folder_name = full_path_parts[-2] # e.g., 'nama_dokumen'
                file_name = os.path.basename(json_path).replace('_o_dt.json', '') # e.g., 'nama_dokumen'
                
                # Asumsikan file JSON memiliki struktur dengan 'ocr_details'
                ocr_details = data.get('ocr_details', [])
                if not ocr_details:
                    continue

                # Kelompokkan data berdasarkan nomor halaman
                pages_data = {}
                for item in ocr_details:
                    page_num = item.get('page')
                    if page_num is not None:
                        if page_num not in pages_data:
                            pages_data[page_num] = []
                        pages_data[page_num].append(item)
                
                # Simpan ke dalam self.corpus
                for page_num, items in pages_data.items():
                    # Gunakan nama file asli sebagai kunci, bukan nama folder
                    source_file = f"{file_name}.pdf" # Asumsikan sumbernya adalah PDF
                    self.corpus[(source_file, page_num)] = items

            except Exception as e:
                print(f"[ERROR] Failed to load or parse {json_path}: {e}")

    def get_page_text(self, source_file, page_num):
        """Menggabungkan semua teks dari halaman tertentu."""
        key = (source_file, page_num)
        if key in self.corpus:
            return "\n".join([item.get('text', '') for item in self.corpus[key]])
        return ""

# Contoh penggunaan (opsional, untuk testing)
if __name__ == '__main__':
    # Ganti dengan path output Anda
    loader = JSONCorpusLoader("/home/aam-imanullah/milvus/output")
    if loader.corpus:
        first_key = list(loader.corpus.keys())[0]
        print(f"Data for {first_key}:")
        # print(loader.corpus[first_key])