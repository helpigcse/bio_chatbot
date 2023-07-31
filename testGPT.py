from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
# import os
# import argparse
import time
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

load_dotenv()

embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"

model_type = "GPT4All"
model_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
model_n_ctx = 1000
model_n_batch = 8
target_source_chunks = 4

model_id = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200
)

local_llm = HuggingFacePipeline(pipeline=pipe)

from constants import CHROMA_SETTINGS

# def main():
# Parse the command line arguments
# args = parse_arguments()
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [StreamingStdOutCallbackHandler()] #[]
# Prepare the LLM
# match model_type:
#     case "LlamaCpp":
#         pass
#         # llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#     case "GPT4All":
#         pass
#         # llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#     case _default:
#         # raise exception if model_type is not supported
#         raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
qa = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=retriever, return_source_documents=False) #not args.hide_source)
# Interactive questions and answers
# while True:
#     query = input("\nEnter a query: ")
#     if query == "exit":
#         break
#     if query.strip() == "":
#         continue

#     # Get the answer from the chain
#     start = time.time()
#     res = qa(query)
#     answer, docs = res['result'], [] #if args.hide_source else res['source_documents']
#     end = time.time()

#     # Print the result
#     print("\n\n> Question:")
#     print(query)
#     print(f"\n> Answer (took {round(end - start, 2)} s.):")
#     print(answer)

    # Print the relevant sources used for the answer
    #for document in docs:
    #    print("\n> " + document.metadata["source"] + ":")
    #    print(document.page_content)   

st.set_page_config(page_title="HelpIGCSE Biology Chatbot")

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Welcome to the HelpIGCSE Biology chatbot! How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(query):
    res = qa(query)
    answer, docs = res['result'], []
    return answer


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        answer = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            st.markdown(
                f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                f"<span style='font-weight: bold;'>Edgebricks:</span><br>{st.session_state['generated'][i]}</div>",
                unsafe_allow_html=True
            )
    if st.button("Reset"):
        st.session_state['generated'] = ["I'm HelpIGCSE's Biology chatbot, how may I help you?"]
        st.session_state['past'] = ['Hi!']

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
#                                                  'using the power of LLMs.')
#     parser.add_argument("--hide-source", "-S", action='store_true',
#                         help='Use this flag to disable printing of source documents used for answers.')

#     parser.add_argument("--mute-stream", "-M",
#                         action='store_true',
#                         help='Use this flag to disable the streaming StdOut callback for LLMs.')

#     return parser.parse_args()


# if __name__ == "__main__":
#     main()
