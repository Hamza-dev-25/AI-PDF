import streamlit as st
from Backend.main import ask_question
from Backend.Database import create_vectorstore

st.set_page_config(
    page_title="AI PDF Study Assistant",
    page_icon="📚"
)

st.title("AI PDF Study Assistant")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_path = "uploaded.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    vector_store = create_vectorstore(pdf_path)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    st.success("PDF uploaded successfully. You can now ask questions.")

    question = st.chat_input(
        "Ask a question about the PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                answer, sources = ask_question(
                    question,
                    retriever
                )

            st.write(answer)

            st.markdown("### Sources")

            pages = set()

            for doc in sources:
                page = doc.metadata.get("page")

                if page is not None:
                    pages.add(page + 1)

            for page in sorted(pages):
                st.write(f"Page {page}")