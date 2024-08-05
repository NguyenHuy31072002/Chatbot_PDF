from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Cau hinh
model_file = "/home/huy31072002/Desktop/Chatbot/chatbot/models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "/home/huy31072002/Desktop/Chatbot/chatbot/src/Vec_tor_DB/vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Tao simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Read tu VectorDB
def read_vectors_db():
    # Embeding
    embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True}) 
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db


# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)

#Tao Prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

# Chay cai chain
question = "Các thành phần của báo cáo tài chính ?"
response = llm_chain.invoke({"query": question})
print(response)

# Chay cai chain
question = "Báo cáo tài chính được lập trên giả định gì?"
response = llm_chain.invoke({"query": question})
print(response)