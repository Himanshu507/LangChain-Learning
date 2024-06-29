from chromadb import Settings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from utils import Utils

# Get embeddings with the device type "mps"
embeddings = Utils.get_embeddings("mps")

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Split text
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=0,
)

# Load and split documents
loader = TextLoader('text.txt')
docs = loader.load_and_split(text_splitter)

# Create Chroma vector store with embeddings
db = Chroma(embedding_function=embeddings, persist_directory="db", client_settings=CHROMA_SETTINGS)


# Define a function to add documents in smaller batches
def add_documents_in_batches(docs, batch_size):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        db.add_documents(batch)


# Add documents in batches of 100 (or any size less than 166)
add_documents_in_batches(docs, batch_size=100)
print("Chrome DB created Successfully")
#
# # Perform similarity search
# results = db.similarity_search_with_score("What happened on 15th May 2023?")
#
# # Print results
# for result in results:
#     print("\n")
#     print(result[1])
#     print(result[0].page_content)
