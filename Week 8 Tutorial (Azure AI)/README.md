# Tuesday 24/09/3034

## 📘 Intro

In this code repo and lecture slides we will cover: 

- Azure OpenAI API Access
- Azure AI Search
- Anatomy of RAG 
- optional: search data ingestion using a notebook 
- Code Samples: chat completion. embeddings and similarity, search methods 
- chat UI with gradio and streamlit
- putting it all together Chat UI for RAG with gradio and streamlit 

Lecture material: <img src="https://img.icons8.com/color/48/000000/pdf.png" alt="PDF icon" width="24" height="24"/> [Building LLM Apps 2024.09.22](<UTSMDSI-Azure OpenAI Service - LLM - 2024.09.22.pdf>)

## 🐍 Installing openao python SDK 

You can use the `venv` module (or conda) to create a python env

Create a Virtual Environment with venv:  `python -m venv .venv` 

Activate the Environment: 

- On Windows: `.venv\Scripts\activate`
- On macOS/Linux: `source .venv/bin/activate`

install openai using pip: `pip install openai`

Alternatively, create a requirements.txt file with all dependencies: `pip install –r requirements.txt`

Sample `requirements.txt` file: 

```
gradio
streamlit
openai
python-dotenv>=1.0.0
azure-search-documents==11.6.0b1
azure-identity
```

## 🤖 How to use AzureOpenAI 

Import Azure OpenAI from the openai library:
```python
from openai import AzureOpenAI
```

To create vector embeddings import the embeddings module: 

```python 
from openai import embeddings
```

Create and Azure OpenAI client: 

```python 
client = AzureOpenAI(
    azure_endpoint=api_endpoint,
    api_key=api_key,
    api_version=api_version,
)
```

### 💬 Chat Completions
You will use the chat completions API to interact with the openai models:
https://platform.openai.com/docs/api-reference/chat/create?lang=python 

```python
response = client.chat.completions.create(
    model=deployment_name_chat,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
)

print(response.choices[0].message.content)
```

### 📊 Vector Embeddings
Using the client embeddings function to create vector embeddings: 

```python
embedding = client.embeddings.create(
    model=deployment_name_embeddings,
    input="Hello, world!"
)

for e in embedding:
    print(e)
```

## 🖥️ Using Streamlit with VS Code

Install streamlit and demo app: https://docs.streamlit.io/get-started/installation/command-line 
Streamlit chat app tutorial: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps 

To debug `streamlit` apps, add the follow to your `launch.json` file: 

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: streamlit Module",
            "type": "debugpy",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "${file}"],
        }
    ]
}
```

