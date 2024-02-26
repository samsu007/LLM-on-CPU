from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.chroma import Chroma


def get_llm_model(model_config, model_name, model_file):
    if len(model_file) >= 0:
        model_file = None
    model = CTransformers(
        model=model_name,
        model_file=model_file,
        config=model_config
    )

    return model


def get_model_prompt(template, input_variables, template_format):
    print(template, input_variables, template_format)
    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables,
        template_format=template_format
    )

    return prompt


def get_model_chain(chain_type, llm, prompt):
    if chain_type == "Conversation Chain":
        chain = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(),
            prompt=prompt
        )
    else:
        chain = LLMChain(
            llm=llm,
            prompt=prompt
        )
    return chain


def get_model_response(chain, chain_input):
    chain_response = chain.run(chain_input)
    return chain_response


def apply_text_splitter(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def load_embedding():
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def convert_files_to_vector(db_name, file):
    loaders = PyPDFLoader(file)
    data = loaders.load()
    documents = apply_text_splitter(data)
    vector_store = Chroma.from_documents(
        documents,
        embedding=load_embedding(),
        persist_directory=f"./{db_name}"
    )
    vector_store.persist()


def process_data(db_name, files):
    for file in files:
        convert_files_to_vector(db_name, file)
    return True


def get_rag_model_response(db_name, context_length, model_name, model_file, rag_model_input_msg):
    vector_store = Chroma(
        persist_directory=f"./{db_name}",
        embedding_function=load_embedding()
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    model_config = {
        "context_length": int(context_length)
    }
    llm_model = get_llm_model(model_config, model_name, model_file)
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff"
    )

    response = retrieval_qa_chain(rag_model_input_msg)

    return response
