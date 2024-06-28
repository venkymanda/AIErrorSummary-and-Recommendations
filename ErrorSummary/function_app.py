import azure.functions as func
import datetime
import json
import logging
import openai
import azure.functions as func
import json
import os
import pinecone


from pinecone import ServerlessSpec
from langchain import LLMChain, PromptTemplate, OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
OPENAI_BASE_URL=os.getenv('OPENAI_BASE_URL')

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY
openai.base_url=OPENAI_BASE_URL

# Set up Pinecone API key and environment
# configure client
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set your API keys

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'


spec = ServerlessSpec(cloud=cloud, region=region)
# we create a new index

error_index = PINECONE_INDEX_NAME
if error_index not in pc.list_indexes().names():
    pc.create_index(
        "error-summary-recommendations-small",
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=spec
    )
    
index = pc.Index(error_index)

# Define the embedding function (you need to set up your embed function)
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
)

# Initialize Pinecone vector store
from langchain.vectorstores import Pinecone
errorlog_vectorstore = Pinecone(index, embed.embed_query, "text")

# Define a prompt template for the LLM
prompt_template = """
Given the following document content, extract the error message, recommendation, and logic app name. Additionally, summarize the error message.

Document: {document}

Please provide the output in the following JSON format:

{{
    "Error Summary": "<summary_of_error_message>",
    "Recommendation": "<extracted_recommendation>",
    "Logic App Name": "<extracted_logic_app_name>"
}}
"""

import langchain,langchain_core
from langchain.chat_models import ChatOpenAI

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
llm = ChatOpenAI(temperature=0.0, model=llm_model,base_url=OPENAI_BASE_URL)
from langchain.chains import LLMChain




# Create a chain
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["document"])
)


app = func.FunctionApp()

@app.route(route="GenerateAIResponse", auth_level=func.AuthLevel.ANONYMOUS)
def GenerateAIResponse(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    query = req.params.get('errormessage')
    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            query = req_body.get('errormessage')
     # Perform similarity search on Pinecone
    if query:
        results = errorlog_vectorstore.similarity_search(query, k=1)
        # Process the documents
        responses = []
        for doc in results:
            response = llm_chain.run(document=doc.page_content)
            try:
                response_json = json.loads(response)
                responses.append(response_json)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON response: {response}")
                return func.HttpResponse(f"Failed to parse JSON response: {response}", status_code=500)

        return func.HttpResponse(json.dumps(response_json), status_code=200, mimetype="application/json")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a Body in the query string or in the request body for a personalized response. from AI ",
             status_code=200
        )

    

    