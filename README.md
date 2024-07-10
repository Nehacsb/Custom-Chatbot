# Custom-Chatbot

The implementation: 
https://colab.research.google.com/drive/1w0bEtde8Wi2rUkCUV7hUKvzJQrmvhd0y?usp=sharing


#### in google colab in the left sidebar  right click to create a new folder ( rename it to "Data")  and in it upload the pdf, this will be the source from which we  perform the  queries.
## Installed Libraries:
      !pip install -q pypdf
      !pip install -q python-dotenv
      !pip install -q transformers
      !CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
      !pip install -q llama-index
      !pip install --upgrade llama-index
      !pip install llama-index-core
      !pip install llama-index-llms-openai
      !pip install llama-index-llms-replicate
      !pip install llama-index-embeddings-huggingface

pypdf: Library for working with PDF files in Python.
<br>
python-dotenv: For loading environment variables from a .env file.
<br>
transformers: HuggingFace library for natural language processing models.
<br>
llama-cpp-python: A package for working with Llama models using Python.
<br>
llama-index and its extensions (llama-index-core, llama-index-llms-openai, llama-index-llms-replicate, llama-index-embeddings-huggingface): Libraries for creating and managing indexes, querying them, and integrating different language models and embeddings.

## Logging Setup:
      import logging
      import sys
      logging.basicConfig(stream=sys.stdout, level=logging.INFO)
      logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

      
Sets up logging to output logs to the console with INFO level.

## Load Documents
      from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
      documents = SimpleDirectoryReader("/content/Data/").load_data()
      SimpleDirectoryReader: Reads documents from the specified directory (/content/Data/).
      load_data(): Loads the documents for further processing.
## LlamaCPP Model Setup 
      import torch
      !pip install llama_index.llms.llama_cpp
      from llama_index.llms.llama_cpp import LlamaCPP
      from llama_index.llms.llama_cpp.llama_utils import (
        messages_to_prompt,
        completion_to_prompt,
        )

        llm = LlamaCPP(
            model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )
torch: PyTorch library for tensor computations and GPU acceleration.
<br>
LlamaCPP: Configures a Llama model with specified parameters:
<br>
model_url: URL to download the model.
<br>
model_path: Optional local path to a pre-downloaded model.
<br>
temperature: Controls the randomness of the model's output.
<br>
max_new_tokens: Maximum number of new tokens to generate.
<br>
context_window: The window size for context.
<br>
generate_kwargs: Additional arguments for generation.
<br>
model_kwargs: Additional arguments for model initialization.
<br>
messages_to_prompt and completion_to_prompt: Utilities to convert messages to prompts and vice versa.
<br>
verbose=True: Enables verbose logging.
<br>

#### Simple Query to Test the Model:

    prompt = "What is the capital of France?"
    response = llm.complete(prompt)
    print(response)

    
prompt: Input query to test the model.
<br>
llm.complete(prompt): Generates a response to the prompt. 


response generated: 
   

llama_print_timings:        load time =   28893.72 ms
llama_print_timings:      sample time =       6.06 ms /     8 runs   (    0.76 ms per token,  1319.26 tokens per second)
llama_print_timings: prompt eval time =   28892.46 ms /    71 tokens (  406.94 ms per token,     2.46 tokens per second)
llama_print_timings:        eval time =    5066.55 ms /     7 runs   (  723.79 ms per token,     1.38 tokens per second)
llama_print_timings:       total time =   33975.54 ms /    78 tokens
 The capital of France is Paris. 



## Install Embedding and Langchain Libraries:


    !pip -q install sentence-transformers
    !pip install langchain
    !pip install langchain_huggingface

    
sentence-transformers: Library for sentence embeddings.
<br>
langchain and langchain_huggingface: Libraries for chaining language model operations and integrating HuggingFace embeddings.


## Set up Embeddings:
  from langchain_huggingface.embeddings import HuggingFaceEmbeddings
  !pip install llama-index-embeddings-huggingface
  from llama_index.core import ServiceContext
  from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  from llama_index.core import Settings

  service_context = ServiceContext.from_defaults(
      chunk_size=256,
      llm=llm,
      embed_model=HuggingFaceEmbedding(
          model_name="BAAI/bge-small-en-v1.5"
      )
  )



HuggingFaceEmbeddings: Utilizes HuggingFace models for generating embeddings.
<br>
ServiceContext: Context for the service including LLM and embedding model configurations.
<br>
chunk_size: Size of chunks for processing.
<br>
llm: Language model instance.
embed_model: Embedding model instance using a specific HuggingFace model (BAAI/bge-small-en-v1.5).

## Create Index and Query Engine:


      from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
      index = VectorStoreIndex.from_documents(documents, service_context=service_context)

      
VectorStoreIndex: Index for storing document vectors.
<br>
from_documents(documents, service_context=service_context): Creates an index from the loaded documents using the specified service context.

## Query the Index:
query = input()
response = query_engine.query(query)
print(response)


query: Takes user input as a query.
<br>
query_engine.query(query): Queries the index and retrieves the response.
<br>

### Query Asked 
How is the academic system designed?
<br>

response received: 
llama_print_timings:        load time =   42007.07 ms
llama_print_timings:      sample time =     171.27 ms /   246 runs   (    0.70 ms per token,  1436.35 tokens per second)
llama_print_timings: prompt eval time =  322390.77 ms /   757 tokens (  425.88 ms per token,     2.35 tokens per second)
llama_print_timings:        eval time =  215842.29 ms /   245 runs   (  880.99 ms per token,     1.14 tokens per second)
llama_print_timings:       total time =  538747.11 ms /  1002 tokens
 The academic system at IIT Ropar is designed to provide a science-based engineering education with the goal of producing quality engineer-scientists. The curriculum provides broad-based knowledge and simultaneously builds a temper for life-long learning and exploring. The undergraduate program begins with a set of science and general engineering courses that are reflected in the course plan for the first year. These courses provide a foundation for further discipline-specific courses. The medium of instruction at IIT Ropar is English. The current academic year began in October 2022 and is expected to end in July 2023. The detailed schedule of activities and academic deadlines is given in the semester schedule and is available on the website link: ... . Education at the Institute is organized around a semester-based credit system of study. The prominent feature of the credits system is a process of continuous evaluation of a student's performance/progress, and the flexibility to allow a student to progress at an optimal pace suited to his/her ability or convenience. This feature is subject to the fulfillment of the minimum requirement for continuation.








