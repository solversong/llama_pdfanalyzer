import torch 
import streamlit as st
from pathlib import Path

# transformer classes for generation
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# prompt wrapper for llama index
from llama_index.core.prompts.prompts import SimpleInputPrompt

# deps to load documents
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader

# pip install llama-index-llms-huggingface
from llama_index.llms.huggingface import HuggingFaceLLM

# pip install llama-index-embeddings-langchain
from llama_index.embeddings.langchain import LangchainEmbedding

# bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# bring in stuff to change service context -> new version
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

# huggingface model name
model_name = ""

# huggingface varification code
auth_token = ""

# document you want analyze
document_path = "./data/VLM-Survey.pdf"

# prevent streamlit to reload resources into memory
@st.cache_resource
def get_tokenizer_model():
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir="./model/",
                                              token=auth_token)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                             llm_int8_threshold=200.0)
    # create model
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir="./model/",
                                                 token=auth_token,
                                                 torch_dtype=torch.float16,
                                                 quantization_config=quantization_config)
    return tokenizer, model

tokenizer, model = get_tokenizer_model()

system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,
while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous,
or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to question, please don't share false information.

Your goal is to provide answers relating to the paper's core contents.<</SYS>>
"""

# throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")
query_wrapper_prompt.format(query_str="hello")

llm = HuggingFaceLLM(context_window=4096,
                     max_new_tokens=256,
                     system_prompt=system_prompt,
                     query_wrapper_prompt=query_wrapper_prompt,
                     model_name=model_name,
                     tokenizer_name=model_name,
                     model=model,
                     tokenizer=tokenizer)

embeddings = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

Settings.llm = llm
Settings.embed_model = embeddings
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

loader = PyMuPDFReader()
documents = loader.load(file_path=Path(document_path), metadata=True)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

st.title("LLaMA PDF Analysis")

prompt = st.text_input("Analyze Whatever You Want Here")

if prompt:
    response = query_engine.query(prompt)
    st.write(response.response)
    
    with st.expander("Response Object"):
        st.write(response)
    
    with st.expander("Source Text"):
        st.write(response.get_formatted_sources())