import gradio as gr

from cpu_llm.cpu_llm import respond

with gr.Blocks(title="CPU LLM") as demo:
    with gr.Row():
        model_name = gr.Textbox(label="Model Name")
        model_file = gr.Textbox(label="Model File")
        context_length = gr.Textbox(label="Context Length")
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            model_chain_type = gr.Dropdown(
                ["LLM Chain"], label="Chain Type", interactive=True
            )
            model_inst_template = gr.TextArea(label="Instruction Template")
            model_input_variables = gr.TextArea(label="Input Variables", value="[]", interactive=True)
        with gr.Column(scale=2, min_width=600):
            chatbot = gr.Chatbot()
            model_input_msg = gr.Textbox(placeholder="Input your message....")
            clear = gr.ClearButton([model_input_msg, chatbot])
            model_input_msg.submit(
                respond, [model_name, model_file, context_length, model_chain_type, model_inst_template,
                          model_input_variables, model_input_msg, chatbot],
                [model_input_msg, chatbot]
            )

demo.launch()
