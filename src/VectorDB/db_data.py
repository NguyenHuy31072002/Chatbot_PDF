from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import SentenceTransformerEmbeddings


# Khởi tạo embeddings
embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True})

vector_db_path = "/home/huy31072002/Desktop/Chatbot/chatbot/src/Vec_tor_DB/vectorstores/db_faiss" 
db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

# Truy xuất embeddings và các dữ liệu khác
index = db.index
vectors = index.reconstruct_n(0, index.ntotal)


for vector in vectors:
    print(vector)


# Kiểm tra các thuộc tính và phương thức của đối tượng db
print(dir(db))


query = "Báo cáo tài chính được lập trên giả định ?"
results = db.similarity_search(query, k=3)  # k là số lượng kết quả bạn muốn trả về

for result in results:
    print(result.page_content)

