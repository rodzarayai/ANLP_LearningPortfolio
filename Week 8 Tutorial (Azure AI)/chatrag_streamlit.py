# %%
from openai import OpenAI
import streamlit as st
from openai import AzureOpenAI

from dotenv import load_dotenv
import os

from helpers.rag import get_system_message, GROUNDED_PROMPT, stream_augmented_generation


# %%

load_dotenv(override=True) # take environment variables from .env.


# Set the API key and endpoint
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  # e.g., "https://<your-resource-name>.openai.azure.com/"
api_type = 'azure'
api_version = '2023-05-15'  # Use the appropriate API version

# Define the deployment name
deployment_name_chat = 'gpt-4o-global'
deployment_name_embeddings = 'text-embedding-ada-002'

# %%
# Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=api_endpoint,
    api_key=api_key,
    api_version=api_version,
)

# %%
# initialise session state for the app


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = deployment_name_chat

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'grounded_prompt' not in st.session_state:
    st.session_state.grounded_prompt = GROUNDED_PROMPT

def print_messages(messages=[]):
    for index, message in enumerate(messages):
        print(f"{index}: {message['role']} - {message['content']}")

# %%
def stream_chat(messages, model_name='gpt-4o-global'):

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
        stream=True
    )

    return stream

# %%

# Set the page configuration to collapse the sidebar by default
st.set_page_config(
    page_title="RAG Chatbot",
    # layout="wide",
    initial_sidebar_state="collapsed"
)

# Add a sidebar with a text box for the system prompt
system_prompt = st.sidebar.text_area("System Prompt", value=st.session_state.grounded_prompt, height=650)
if system_prompt != st.session_state.grounded_prompt:
    st.session_state.grounded_prompt = system_prompt


for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):


    with st.chat_message("user"):
        st.markdown(prompt)

    grounded_input = get_system_message(prompt,metaprompt=system_prompt)
    st.session_state.messages.append({"role": "system", "content": grounded_input})
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("############")
    # print(grounded_input)
    print_messages(st.session_state.messages)

    with st.chat_message("assistant"):
        # stream = stream_chat(st.session_state.messages, model_name=st.session_state["openai_model"])
        stream = stream_augmented_generation(st.session_state.messages, model_name=st.session_state["openai_model"])
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

# %%



