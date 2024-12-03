import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter



# Get current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

#Define database directory and persistence directory
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_booking")

#Inicialize URL for web-based booking API
url= ["https://www.wikipedia.com/"]


# Create a loader for web content
loader = WebBaseLoader(url)
documents = loader.load()

# CharacterTextSplitter splits the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Initialize Chroma and embeddings for retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})
if os.path.exists(persistent_directory):
    db=Chroma.from_documents(docs, embeddings,persist_directory=persistent_directory)
else:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Initialize the retriever
retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 3})

#Initialize the LLM
llm = ChatOllama(model = "llama3.1", temperature = 0.8)

# Define system prompt for contextualizing the question
contextualize_q_system_prompt = ("Given a chat history and the latest user question which might reference context in the chat history,"
                                " formulate a standalone question which can be understood without the chat history. "
                                " Do NOT answer the question, just reformulate it if needed and otherwise return it as is.")
# Create prompt with system message, chat history, and user input
contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
)
# Conversation loop
def chat():

    print("\n\nStart chatting with the AI! Type 'exit' to end the conversation.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        relevant_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        result = llm.invoke(f"Context: {context}\nQuestion: {query}")

        print(f"AI Response: ", result.content)


if __name__ == "__main__":
    chat()