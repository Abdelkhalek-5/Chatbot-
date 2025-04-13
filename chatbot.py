import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ✅ Load PDF documents (limited for memory safety)
def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    documents = documents[:50]  # 🔒 Limit to 50 docs (can increase later)
    print(f"✅ Loaded {len(documents)} PDF documents.")
    return documents

# ✅ Split documents into small chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"✅ Split into {len(text_chunks)} text chunks.")
    return text_chunks

# ✅ Setup ChromaDB (in-memory for safety)
def setup_chromadb(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 🔁 In-memory only — no persist_directory or .persist()
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )

    print("🧠 Using in-memory ChromaDB (not saved to disk).")
    return vectorstore

# ✅ Setup LLaMA model
def setup_llama(vectorstore):
    prompt_template = """
    Use the summarized context to answer concisely.

    Context: {context}
    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={
            "max_new_tokens": 64,  # 🔽 Lowered to save RAM
            "temperature": 0.7,
            "batch_size": 4         # 🔽 Lower batch size to avoid segfaults
        }
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa

# ✅ Load all models and vector DB
def load_models_and_data():
    extracted_data = load_pdf("data/")
    text_chunks = text_split(extracted_data)
    vectorstore = setup_chromadb(text_chunks)
    qa = setup_llama(vectorstore)
    return qa

# ✅ Async query call
async def query_qa(user_input, qa):
    user_input = user_input[:150]  # Trim long queries
    result = await qa.ainvoke({"query": user_input})
    return result["result"]

# ✅ Chat loop
def main():
    print("\n🔄 Loading models and data...")
    qa = load_models_and_data()
    print("✅ Models and data loaded successfully.\n")

    print("💬 Fast Cancer Chatbot (type 'exit' to quit)")

    while True:
        user_input = input("\n❓ You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        print("🤖 Thinking...")
        result = asyncio.run(query_qa(user_input, qa))
        print(f"\n📝 Answer: {result}")

# ✅ Entry point
if __name__ == "__main__":
    main()
