import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


print("Loading website content...")
loader = WebBaseLoader(web_paths=("https://home-care-medics.org/",))
docs = loader.load()
print("Content loaded successfully.")

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"{len(splits)} chunks created.")

print("Creating embeddings and vector store...")
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
print("Vector store created.")

llm = ChatOllama(model="llama3.1")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
If you don't know the answer, just say that you don't know. 
Keep your answers concise and helpful.

<context>
{context}
</context>

Question: {input}
""")

print("Creating retrieval chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("Assistant is ready. Ask a question.\n")

question = "What kind of services does Home Care Medics provide?"
response = retrieval_chain.invoke({"input": question})

print("---")
print(f"Question: {question}")
print(f"Answer: {response['answer']}")
print("---")