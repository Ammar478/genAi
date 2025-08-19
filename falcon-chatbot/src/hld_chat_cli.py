from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from colorama import Fore, Style, init


init(autoreset=True)

faiss_path = "data/hld_faiss_index"
embeddings = OllamaEmbeddings(model="llama3.2:latest")
db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = Ollama(model="gpt-oss:20b")

prompt = PromptTemplate(
    template=(
        "You are a chatbot that only answers based on the given context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer clearly based ONLY on the context. "
        "If the answer is not in the context, say: 'I don't know based on the provided HLD document.'"
    ),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

if __name__ == "__main__":
    print(Fore.CYAN + Style.BRIGHT + "🤖 HLD Chatbot (type 'exit' to quit)\n")
    while True:
        query = input(Fore.GREEN + Style.BRIGHT + "You: " + Style.RESET_ALL)
        if query.lower() in ["exit", "quit"]:
            print(Fore.MAGENTA + "👋 Goodbye!")
            break

        response = qa_chain.invoke({"query": query})
        print(
            Fore.YELLOW + "Bot:" + Style.RESET_ALL,
            Fore.WHITE + response["result"] + "\n",
        )
