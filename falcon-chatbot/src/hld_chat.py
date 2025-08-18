from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings

# load vector db
embedding = OllamaEmbeddings(model="llama3.2:latest")
db = FAISS.load_local("data/vector_db", embedding,allow_dangerous_deserialization=True)

retruever = db.as_retriever(search_type="similarity",search_kwargs={"k": 3})

# create chain
llm = OllamaLLM(model="gpt-oss:20b")

# create prompt
prompt = PromptTemplate.from_template(
    template=(
        "You are a chatbot that only answers based on the given context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer clearly based ONLY on the context. "
        "If the answer is not in the context, say: 'I don't know based on the provided HLD document.'"
    ),
)

# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retruever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# test 
if __name__ == "__main__":
    query = "Who won the world cup 2022?"
    response = qa_chain.invoke({"query": query})
    print("🔎 Question:", query)
    print("💬 Answer:", response["result"])
