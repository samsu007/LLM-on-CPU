from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory


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
