from langchain.chains.retrieval_qa.base import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
from utils import Utils
import warnings
import logging
import atexit


# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress detailed logging from llama_model_loader or other specific libraries
logging.getLogger("llama_model_loader").setLevel(logging.ERROR)


device_type = "mps"  # "mps" for mobile phones, "cpu" for CPU
MODEL_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_BASENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
chat = Utils.load_model(device_type=device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=RedundantFilterRetriever(),
    chain_type="stuff"
)

def cleanup():
    # Add any necessary cleanup code here
    if hasattr(chat, 'close'):
        chat.close()

atexit.register(cleanup)

# result = chain.run("explain the code line by line of this method get_price_history()?")
# print(result)

while True:
    query = input("\n\nEnter your question ('exit' to quit): ")
    if query.lower() == "exit":
        print("Exiting the Q&A loop. Goodbye!")
        break

    result = chain.run(query)
    print(result)

cleanup()
