import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# load document

doc_path = './data/row_docs/hld.txt'
loader = TextLoader(doc_path)
doc = loader.load()

print(f"loaded {len(doc)} pages")

# split document
spiltter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
)

chunks = spiltter.split_documents(doc)
print(f"✂️ Created {len(chunks)} chunks.")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk.page_content}")

# embed document
embeddings = OllamaEmbeddings(model="llama3.2:latest")

# store in FAISS vector db
db = FAISS.from_documents(chunks, embeddings)

# save vector db
db.save_local("data/vector_db")


# search
query = "What is the system architecture?"
results = db.similarity_search(query, k=3)
for result in results:
    print("\n--- Query ---\n" + query)
    print("\n--- Result ---\n" + result.page_content)
