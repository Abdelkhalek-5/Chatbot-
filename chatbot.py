import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ✅ Load PDF documents
def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# ✅ Split text into smaller chunks (Improves speed)
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)  # Reduced chunk size for speed
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# ✅ Setup ChromaDB (Optimized)
def setup_chromadb(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_directory = "D:/Chat/chroma_db"
    os.makedirs(db_directory, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    return vectorstore

# ✅ Initialize LLaMA model (Optimized for speed)
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
        config={"max_new_tokens": 128, "temperature": 0.7, "batch_size": 8}  # Lower max tokens & batch processing
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),  # Only fetch 1 document for speed
        return_source_documents=False,  # Disable sources for extra speed
        chain_type_kwargs={"prompt": PROMPT}  # Pass the prompt template here
    )
    return qa

# ✅ Load models and data (Runs once)
def load_models_and_data():
    extracted_data = load_pdf("data/")  
    text_chunks = text_split(extracted_data)
    vectorstore = setup_chromadb(text_chunks)
    qa = setup_llama(vectorstore)
    return qa

# ✅ Asynchronous query function (Non-blocking)
async def query_qa(user_input, qa):
    user_input = user_input[:150]  # Limit input size
    result = await qa.ainvoke({"query": user_input})  # Use async version
    return result["result"]

# ✅ Main chatbot function
def main():
    print("\n🔄 Loading models and data...")
    qa = load_models_and_data()
    print("✅ Models and data loaded successfully.\n")
    
    print("\n💡 Fast Cancer Chatbot (Type 'exit' to quit)")

    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("⏳ Thinking...")
        result = asyncio.run(query_qa(user_input, qa))  # Async for speed
        print(f"\n📝 Answer: {result}")

# ✅ Run the chatbot
if __name__ == "__main__":
    main()
