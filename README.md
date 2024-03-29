# LLM-on-CPU

#### LLM-on-CPU - Access the Large Language Models using CPU (CTransformers Library) with Interactive UI (Gradio Library)

## Quick Start

### Step 1 - Clone this repository

```bash
git clone https://github.com/samsu007/LLM-on-CPU.git
```

### Step 2 - Install the requirements 

```bash
pip install -r requirements.txt
```

### step 3 - Run the UI

```bash
python run_ui.py
```

### LLM on CPU - UI

#### 1. Chat with LLM

![chat_with_llm_ui.JPG](images/chat_with_llm_ui.JPG)

#### 2. RAG with LLM

![chat_with_llm_ui.JPG](images/rag_with_llm_ui.JPG)

### Sample Inputs Configs (* required)

#### 1. Chat with LLM

| **Input Variables**    | **Sample Input Data**               |
|------------------------|-------------------------------------|
| Model Name*            | TheBloke/Llama-2-7B-Chat-GGML       |
| Model File             | llama-2-7b-chat.ggmlv3.q2_K.bin     | 
| Context Length*        | 512                                 |
| Chain Type*            | LLM Chain                           |
| Instruction Template*  | What is {{data}} ?                  |
| Input Variables*       | ["data"]                            |

#### 2. RAG with LLM

| **Input Variables** | **Sample Input Data**           |
|---------------------|---------------------------------|
| Model Name*         | TheBloke/Llama-2-7B-Chat-GGML   |
| Model File          | llama-2-7b-chat.ggmlv3.q2_K.bin | 
| Context Length*     | 512                             |
| Vector DB Name*     | temp_db                         |
| Upload files [PDF]* | sample_file.pdf                 |

### List of CPU Models - Refer the links below
Models - https://huggingface.co/TheBloke

**Note:** In this app you can use **GGML Models** for running those llm in cpu machine. you can refer those model_name , model_file and other config from above given link. 

## Features
1. The current implementation utilizes the CTransformer Library within Langchain to enable CPU-based large language model (LLM) functionality. 
2. Users can customize configurations and interact with the LLMs.
3. Users can chat with the LLM by providing instruction templates.
4. Enabled the Rag functionalities in the app now user can upload the own documents and ask question within the docs using chatbot
5. We are using ChromaDB for storing the vectors in local path

## Need to be added
1. Need to expand the current LLM Chain integration to incorporate other chains from Langchain for broader functionality.
2. Need to improve code structure and clarity for better maintainability and understanding.
3. Refine the user interface for a more intuitive and user-friendly experience.

## Contributing

We welcome any contributions to our open source project, including new features, improvements to infrastructure, and more comprehensive documentation. 
Please see the [contributing guidelines](contribute.md)