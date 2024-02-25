import json

from cpu_llm.utils import get_llm_model, get_model_prompt, get_model_chain, get_model_response


def init_config(model_name, model_file, context_length, model_chain_type, model_inst_template,
                model_input_variables):
    llm_model_config = {
        "context_length": int(context_length)
    }
    llm_model_name = model_name
    llm_model_file = model_file
    llm_model = get_llm_model(
        model_config=llm_model_config,
        model_name=llm_model_name,
        model_file=llm_model_file
    )
    print("LLM Model : ", llm_model)
    instruction_template = model_inst_template
    model_input_variables = model_input_variables
    model_template_format = "jinja2"

    model_prompt = get_model_prompt(
        template=instruction_template,
        input_variables=model_input_variables,
        template_format=model_template_format
    )

    print("LLM Model Prompt : ", model_prompt)

    model_chain_type = model_chain_type
    llm_model_chain = get_model_chain(
        chain_type=model_chain_type,
        llm=llm_model,
        prompt=model_prompt
    )

    print("LLM Model Chain : ", llm_model_chain)

    return llm_model_chain


def respond(model_name, model_file, context_length, model_chain_type, model_inst_template,
            model_input_variables, model_input_msg, chat_history):
    llm_model_chain = init_config(model_name, model_file, context_length, model_chain_type, model_inst_template,
                                  json.loads(model_input_variables))
    model_chain_input = {
        json.loads(model_input_variables)[-1]: model_input_msg
    }
    print("Model Chain Input : ", model_chain_input)
    response = get_model_response(
        chain=llm_model_chain,
        chain_input=model_chain_input
    )
    chat_history.append((model_input_msg, response))
    return "", chat_history
