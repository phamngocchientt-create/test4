# my-tutor-chatbot/app.py

import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CẤU HÌNH CƠ BẢN ---
# Cần phải đặt biến môi trường GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Lỗi: Vui lòng thiết lập GOOGLE_API_KEY trong Streamlit Secrets hoặc biến môi trường.")
    st.stop()

# Lấy đường dẫn tuyệt đối đến thư mục chứa file app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Nối đường dẫn đó với thư mục "documents"
DOCUMENT_FOLDER = os.path.join(APP_DIR, "documents")


# ---------------------------------------------------------------------
# GIAI ĐOẠN 2 & 3: TẢI, XỬ LÝ, VÀ THIẾT LẬP CHUỖI RAG
# Sử dụng st.cache_resource để chỉ chạy hàm này MỘT LẦN khi ứng dụng khởi động
# ---------------------------------------------------------------------

@st.cache_resource
def setup_rag_system():
    """Tải tài liệu, tạo vector, và thiết lập chuỗi RAG."""
    st.info("Đang khởi tạo Gia sư: Tải tài liệu, tạo embeddings...")

    # 1. Khởi tạo LLM và Embedding Model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Tải tất cả Tài liệu từ thư mục 'documents'
    all_documents = []

    if not os.path.exists(DOCUMENT_FOLDER):
        st.error(f"Thư mục tài liệu '{DOCUMENT_FOLDER}' không tồn tại.")
        st.stop()

    for filename in os.listdir(DOCUMENT_FOLDER):
        file_path = os.path.join(DOCUMENT_FOLDER, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue

        all_documents.extend(loader.load())

    if not all_documents:
        st.warning(f"Không tìm thấy tài liệu nào trong thư mục '{DOCUMENT_FOLDER}'.")
        st.stop()

    # 3. Chia đoạn (Chunking) Tối ưu
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    chunks = text_splitter.split_documents(all_documents)

    # 4. Tạo và Lưu trữ Vector (dùng ChromaDB tạm thời)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # Lấy 4 đoạn liên quan nhất

    # 5. Xây dựng Chuỗi RAG với Prompt Gia sư
    SYSTEM_PROMPT = """
    Bạn là một gia sư chuyên nghiệp, nhiệt tình và kiên nhẫn. Nhiệm vụ của bạn là:
    1. Trả lời câu hỏi của học sinh MỘT CÁCH CHÍNH XÁC, DỰA TRÊN NỘI DUNG TÀI LIỆU được cung cấp bên dưới.
    2. Giải thích các khái niệm một cách dễ hiểu.
    3. Sau mỗi câu trả lời, hãy ĐẶT MỘT CÂU HỎI NGẮN GỌN để kiểm tra sự hiểu biết.

    Nội dung tài liệu tham khảo:
    ----------------
    {context}
    ----------------
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success(f"Khởi tạo hoàn tất. Đã xử lý {len(chunks)} đoạn kiến thức.")
    return retrieval_chain


# --- ỨNG DỤNG STREAMLIT CHÍNH (GIAI ĐOẠN 4) ---

st.title("👨‍🏫 Chatbot Gia Sư Cá Nhân")
st.caption("Sử dụng Gemini API và RAG để trả lời dựa trên tài liệu của bạn.")

# Khởi tạo RAG System (chỉ chạy lần đầu)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = setup_rag_system()

# Lấy chuỗi RAG đã tạo
retrieval_chain = st.session_state.rag_chain

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input của người dùng
if prompt := st.chat_input("Hỏi gia sư của bạn một câu hỏi..."):
    # Thêm tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi RAG Chain để lấy câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Gia sư đang tìm kiếm và trả lời..."):
            try:
                # Gọi chuỗi RAG
                response = retrieval_chain.invoke({"input": prompt})
                answer = response["answer"]

                # Hiển thị câu trả lời và thêm vào lịch sử
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"Có lỗi xảy ra khi gọi API: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
