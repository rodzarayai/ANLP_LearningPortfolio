# %%
from openai import OpenAI
from openai import AzureOpenAI

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import os
import re 

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from azure.search.documents.models import (
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)

# %%

load_dotenv(override=True) # take environment variables from .env.

endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_QUERY_KEY"]) 
index_name = os.environ["AZURE_SEARCH_INDEX"]


# Set the API key and endpoint
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  # e.g., "https://<your-resource-name>.openai.azure.com/"
api_type = 'azure'
api_version = '2023-05-15'  # Use the appropriate API version

# Define the deployment name
deployment_name_chat = 'gpt-4o-global'
deployment_name_embeddings = 'text-embedding-ada-002'


# %%

def get_page_number(chunk_id : str): 
    page_re = r'_pages_(\d+)$'

    match = re.search(page_re, chunk_id)
    if match:
        page_number = match.group(1)
        return page_number

# %%
# Set query parameters for grounding the conversation on your search index


# %%
client = AzureOpenAI(
    azure_endpoint=api_endpoint,
    api_key=api_key,
    api_version=api_version,
)

# %%
search_client = SearchClient(
    endpoint, 
    index_name, 
    credential=credential)



# %%
 # This prompt provides instructions to the model
GROUNDED_PROMPT="""
You are a friendly and knowledgeable assistant who answers questions using only the information provided in the company's HR policy and information documents listed below. Respond to each query in a friendly, concise, and bulleted manner, relying solely on the facts from these sources. If the necessary information is not available in the provided sources, kindly inform the user that you do not know. Do not include any information that is not present in the sources. Always cite the source of the information by mentioning the file name and page number.

Sources:

{sources}

Semantic Answer:

{semantic_response}


 """

SEARCH_INTENT_PROMPT = """
You are asked to read the user query, understand it, and then turn it into a search query.
The search query should be sent to the search engine to find the relevant information.
the search query should be similar to what a user would type into a search engine, and similar in meaning to the provided user query.
the user query is: '{user_query}'
return only the search query text. """

# user_query = input("Enter your question: ")

# %%
#extract user search intent



def extract_search_intent(user_query="Is pregnancy going to be covered by my health plan?"):
    # print(SEARCH_INTENT_PROMPT.format(user_query=user_query))

    response = client.chat.completions.create(
        model=deployment_name_chat,
        messages=[
            {"role": "system", "content": SEARCH_INTENT_PROMPT.format(user_query=user_query)},
        ]
    )

    search_query = response.choices[0].message.content

    # print(search_query)
    return search_query

# %%
 # Retrieve the selected fields from the search index related to the question.

def get_search_results(
        text_query="Is pregnancy going to be covered by my health plan?", 
        search_type="text",
        use_semantic_reranker=True,
        sources_to_include=5):



    vector_query = VectorizableTextQuery(text=text_query, k_nearest_neighbors=1, fields="vector", exhaustive=True)

    results = search_client.search(  
        search_text=text_query,
        vector_queries=[vector_query],
        select=["parent_id", "chunk_id", "chunk", "title"],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name='my-semantic-config',
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=sources_to_include
    )

    semantic_answers = results.get_answers()
    semantic_response = ""
    if semantic_answers:
        for answer in semantic_answers:
            if answer.highlights:
                semantic_response = answer.highlights
                # print(f"Semantic Answer: {answer.highlights}")
            else:
                semantic_response = answer.text
                # print(f"Semantic Answer: {answer.text}")
            # print(f"Semantic Answer: {semantic_response}")
            # print(f"Semantic Answer Score: {answer.score}\n")

    list_of_sources = [f'\nSOURCE NO {index + 1}. {result["title"]}({get_page_number(result["chunk_id"])}): {result["chunk"]} )' for index, result in enumerate(results)]
    # print(list_of_sources)


    joined_sources = "\n".join(list_of_sources)
    # print(joined_sources)

    return joined_sources, semantic_response



def get_system_message(user_query, metaprompt=GROUNDED_PROMPT): 
    # print(f"User Query: {user_query}")
    search_query = extract_search_intent(user_query)
    sources, semantic_response = get_search_results(search_query)

    grounded_input = metaprompt.format(sources=sources, semantic_response=semantic_response)
    return grounded_input


def get_augmented_generation(user_query, metaprompt):

    grounded_input =get_system_message(user_query, metaprompt)
    
    response = client.chat.completions.create(
        model=deployment_name_chat,
        messages=[
            {"role": "system", "content": grounded_input},
            {"role": "user", "content": user_query}
        ]
    )

    # print(user_query)
    # print(search_query)


    final_response = response.choices[0].message.content

    # print(final_response)
    return final_response


def stream_augmented_generation(messages, model_name='gpt-4o-global'):
    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ], 
        stream=True
    )
    
    return stream




