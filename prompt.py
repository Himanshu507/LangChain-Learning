from langchain.chains.retrieval_qa.base import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
from utils import Utils


device_type = "mps"  # "mps" for mobile phones, "cpu" for CPU
MODEL_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_BASENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
chat = Utils.load_model(device_type=device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=RedundantFilterRetriever(),
    chain_type="stuff"
)

result = chain.run("explain the code line by line of this method get_price_history()?")

print(result)
