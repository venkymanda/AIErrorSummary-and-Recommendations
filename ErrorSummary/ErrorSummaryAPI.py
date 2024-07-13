import azure.functions as func
import logging
import json
import os
import openai
from bs4 import BeautifulSoup
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pyodbc

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL')
SQL_CONNECTION_STRING = os.getenv('SQL_CONNECTION_STRING')


# Initialize OpenAI
openai.api_key = OPENAI_API_KEY
openai.base_url = OPENAI_BASE_URL

# Initialize Pinecone and OpenAI Embeddings for similarity search
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


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
    

index = pc.Index(PINECONE_INDEX_NAME)

embed = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
)

# Initialize Pinecone vector store
from langchain.vectorstores import Pinecone
errorlog_vectorstore = Pinecone(index, embed.embed_query, "text")

# Initialize ChatOpenAI
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", openai_api_base=OPENAI_BASE_URL)

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


# Define prompt template for error summary
error_summary_prompt = PromptTemplate(
    input_variables=["document"],
    template=prompt_template
)

# Create LLMChain for error summary
error_summary_chain = LLMChain(llm=llm, prompt=error_summary_prompt)

def extract_html_content(email_body):
    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(email_body, 'html.parser')
    return soup.get_text()

def extract_error_message(text):
    # This is a placeholder. Adjust the regex pattern based on your error message format
    error_pattern = r"'dimensions':\s*\[\s*{\s*'name':\s*'resource_workflowName_s',\s*'value':\s*'([^']+)'\s*}\s*\]"
    match = re.search(error_pattern, text)
    return match.group(1) if match else "No error message found"

def query_log_layer_message(content):
    # Query Azure SQL Database for log layer message
    conn = pyodbc.connect(SQL_CONNECTION_STRING)
    cursor = conn.cursor()
    
    # Adjust this query based on your database schema and how you want to match the content
    query = f"""
    SELECT ErrorMessage
    FROM [dbo].[CommonLoggingTest]
    Where FailedLogicApp ='{content}'
    """
    
    cursor.execute(query, ('%' + content + '%',))
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else "No matching log layer message found"

def query_similar_errors(error_message):
    # Query Pinecone for similar errors
    results = errorlog_vectorstore.similarity_search(error_message, k=1)
    return results[0].page_content if results else "No similar errors found"

def generate_error_summary(results):
    # Generate error summary using the LLMChain
    responses = []
    for doc in results:
            response = error_summary_chain.run(error_message=doc.page_content)
            try:
                response_json = json.loads(response)
                responses.append(response_json)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON response: {response}")
                return responses
    return response_json

def send_email_notification(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = NOTIFICATION_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

app = func.FunctionApp()

@app.route(route="ProcessErrorSummaryEmailAPI", auth_level=func.AuthLevel.ANONYMOUS)
def ProcessErrorEmail(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    try:
        req_body = req.get_json()
        logging.info(req_body)
    except ValueError:
        return func.HttpResponse("Invalid JSON in request body", status_code=400)

    email_body = req_body.get('email_body')
    if not email_body:
        return func.HttpResponse("No email body provided in the request", status_code=400)

    # Process the email
    text_content = extract_html_content(email_body)
    error_message = extract_error_message(text_content)
    
    # Query log layer message from Azure SQL Database
    log_layer_message = query_log_layer_message(error_message)
    
    # Query for similar errors
    similar_error = query_similar_errors(error_message)
    
    # Generate error summary
    error_summary = generate_error_summary(similar_error)
    
    # Prepare notification email
    notification_subject = "Error Alert: New Error Processed"
    notification_body = f"""
    Logic app Name:
    {error_message}

    Log Layer Message:
    {log_layer_message}

    Error Summary:
    {error_summary["Error Summary"]}

    Recommendations:
    {error_summary["Recommendation"]}
    """
    
    # Send notification email
    send_email_notification(notification_subject, notification_body)

    return func.HttpResponse(
        json.dumps({
            "message": "Error processed and notification sent", 
            "error_summary": error_summary,
            "log_layer_message": log_layer_message
        }),
        mimetype="application/json",
        status_code=200
    )