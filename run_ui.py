import gradio as gr

from cpu_llm.cpu_llm import respond, respond_rag

with gr.Blocks(title="CPU LLM") as demo:
    with gr.Tab("Chat with LLM"):
        with gr.Row():
            model_name = gr.Textbox(label="Model Name*")
            model_file = gr.Textbox(label="Model File")
            context_length = gr.Textbox(label="Context Length*")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                model_chain_type = gr.Dropdown(
                    ["LLM Chain"], label="Chain Type*", interactive=True
                )
                model_inst_template = gr.TextArea(label="Instruction Template*")
                model_input_variables = gr.TextArea(label="Input Variables*", value="[]", interactive=True)
            with gr.Column(scale=2, min_width=600):
                chatbot = gr.Chatbot()
                model_input_msg = gr.Textbox(placeholder="Input your message....")
                clear = gr.ClearButton([model_input_msg, chatbot])
                model_input_msg.submit(
                    respond, [model_name, model_file, context_length, model_chain_type, model_inst_template,
                              model_input_variables, model_input_msg, chatbot],
                    [model_input_msg, chatbot]
                )
    with gr.Tab("RAG with LLM"):
        with gr.Row():
            model_name = gr.Textbox(label="Model Name*")
            model_file = gr.Textbox(label="Model File")
            context_length = gr.Textbox(label="Context Length*")
            db_name = gr.Textbox(label="Vector DB Name*")
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                files = gr.Files(label="upload files [pdf] *")
            with gr.Column(scale=2, min_width=600):
                rag_chatbot = gr.Chatbot()
                rag_model_input_msg = gr.Textbox(placeholder="Input your message....")
                clear_rag = gr.ClearButton([rag_model_input_msg, rag_chatbot])
                rag_model_input_msg.submit(
                    respond_rag, [files, context_length, model_name, model_file, db_name, rag_model_input_msg, rag_chatbot],
                    [rag_model_input_msg, rag_chatbot]
                )
demo.launch()
