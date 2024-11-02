import gradio as gr
import torch
from transformers import RagRetriever, RagSequenceForGeneration

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "./sample/my_knowledge_dataset"
index_path = "./sample/my_knowledge_dataset_hnsw_index.faiss"

retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom",
                                            passages_path = dataset_path,
                                            index_path = index_path,
                                            n_docs = 5)
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq', retriever=retriever)
rag_model.retriever.init_retrieval()
rag_model.to(device)

def strip_title(title):
    if title.startswith('"'):
        title = title[1:]
    if title.endswith('"'):
        title = title[:-1]
    return title

def retrieved_info(query, rag_model = rag_model):
    # Tokenize query
    retriever_input_ids = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
        [query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to(device)

    # Retrieve documents
    question_enc_outputs = rag_model.rag.question_encoder(retriever_input_ids)
    question_enc_pool_output = question_enc_outputs[0]

    result = rag_model.retriever(
        retriever_input_ids,
        question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
        prefix=rag_model.rag.generator.config.prefix,
        n_docs=rag_model.config.n_docs,
        return_tensors="pt",
    )

    # Display retrieved documents including URLs
    all_docs = rag_model.retriever.index.get_doc_dicts(result.doc_ids)
    retrieved_context = []
    for docs in all_docs:
        titles = [strip_title(title) for title in docs["title"]]
        texts = docs["text"]
        for title, text in zip(titles, texts):
            retrieved_context.append(f"{title}: {text}")

    answer = retrieved_context
    return answer

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens ,
    temperature,
    top_p,
):
    
    if message:  # If there's a user query
        response = retrieved_info(message)  # Get the answer from local FAISS and Q&A model
        return response[0]

    # In case no message, return an empty string
    return ""


# Custom title and description
title = "🧠 Welcome to Your AI Knowledge Assistant"
description = """
HI!!, I am your loyal assistant, My functionality is based on RAG model, I retrieves relevant information and provide answers based on that. Ask me any question, and let me assist you.
My capabilities are limited because I am still in development phase. I will do my best to assist you. SOOO LET'S BEGGINNNN......
"""

demo = gr.ChatInterface(
    respond,
    type = 'messages',
    additional_inputs=[
        gr.Textbox(value="You are a helpful and friendly assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    title=title,
    description=description,
    submit_btn = True,
    textbox=gr.Textbox(placeholder=["'What is the future of AI?' or 'App Development'"]),
    examples=[["✨Future of AI"], ["📱App Development"]],
    #example_icons=["🤖", "📱"],
    theme="compact",
)

if __name__ == "__main__":
    demo.launch(share = True )
