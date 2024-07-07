from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings, \
    HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, GenerationConfig
from load_models import load_quantized_model_gguf_ggml, load_quantized_model_awq, load_quantized_model_qptq, \
    load_full_model, MAX_NEW_TOKENS
import warnings
import logging
import atexit


# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress detailed logging from llama_model_loader or other specific libraries
logging.getLogger("llama_model_loader").setLevel(logging.ERROR)

class Utils:
    EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

    @staticmethod
    def get_embeddings(device_type="cuda"):
        if "instructor" in Utils.EMBEDDING_MODEL_NAME:
            from InstructorEmbedding import INSTRUCTOR
            return HuggingFaceInstructEmbeddings(
                model_name=Utils.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device_type},
                embed_instruction="Represent the document for retrieval:",
                query_instruction="Represent the question for retrieving supporting documents:",
            )

        elif "bge" in Utils.EMBEDDING_MODEL_NAME:
            return HuggingFaceBgeEmbeddings(
                model_name=Utils.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device_type},
                query_instruction="Represent this sentence for searching relevant passages:",
            )

        else:
            return HuggingFaceEmbeddings(
                model_name=Utils.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device_type},
            )

    @staticmethod
    def load_model(device_type, model_id, model_basename=None):
        """
        Select a model for text generation using the HuggingFace library.
        If you are running this for the first time, it will download a model for you.
        subsequent runs will use the model from the disk.

        Args:
            device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
            model_id (str): Identifier of the model to load from HuggingFace's model hub.
            model_basename (str, optional): Basename of the model if using quantized models.
                Defaults to None.

        Returns:
            HuggingFacePipeline: A pipeline object for text generation using the loaded model.

        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        print(f"Loading Model: {model_id}, on: {device_type}")
        print("This action can take a few minutes!")

        if model_basename is not None:
            if ".gguf" in model_basename.lower():
                llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type)
                return llm
            elif ".ggml" in model_basename.lower():
                model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type)
            elif ".awq" in model_basename.lower():
                model, tokenizer = load_quantized_model_awq(model_id)
            else:
                model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type)
        else:
            model, tokenizer = load_full_model(model_id, model_basename, device_type)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)
        # see here for details:
        # https://huggingface.co/docs/transformers/
        # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_NEW_TOKENS,
            temperature=0.2,
            # top_p=0.95,
            repetition_penalty=1.15,
            generation_config=generation_config,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        print("Local LLM Loaded")

        return local_llm