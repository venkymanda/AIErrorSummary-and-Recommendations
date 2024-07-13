import azure.functions as func
import logging
import json
import os
import openai
from bs4 import BeautifulSoup
import datetime
from datetime import datetime
import json
import logging
import pinecone
from pinecone import ServerlessSpec
from langchain import LLMChain, PromptTemplate, OpenAI
import langchain,langchain_core
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import re
from langchain.embeddings.openai import OpenAIEmbeddings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pyodbc
from O365 import Account, Message
from O365.connection import Connection
from O365 import FileSystemTokenBackend


# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL')
SQL_CONNECTION_STRING = os.getenv('SQL_CONNECTION_STRING')

# Office 365 OAuth credentials
CLIENT_ID = os.getenv('OFFICE365_CLIENT_ID')
CLIENT_SECRET = os.getenv('OFFICE365_CLIENT_SECRET')
TENANT_ID = os.getenv('OFFICE365_TENANT_ID')


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
Given the following document content, extract the error message, recommendation, and logic app name. Additionally, summarize the error message.If Error Message is not Present in Knowldege base try to summarize and give recommendation 

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
    error_pattern = r'"name"\s*:\s*"resource_workflowName_s",\s*"value"\s*:\s*"([^"]+)"'
    #error_pattern =  r'['"](name)['"]\s*:\s*['"](resource_workflowName_s)['"]\s*,\s*['"](value)['"]\s*:\s*['"](.*)['"]'
    test=text.replace("'", '"')
    match = re.search(error_pattern, text.replace("'", '"'))
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
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else "No matching log layer message found"

def query_similar_errors(error_message):
    # Query Pinecone for similar errors
    results = errorlog_vectorstore.similarity_search(error_message, k=1)
     # Check the condition and return the appropriate result
    if results and error_message != "No matching log layer message found":
        return results
    else:
        error_response = [
            ErrorDocumentFormat(
                page_content= "Error: Error Log is Not found in Knowledge Base\nRecommendation: Try to Look at the error by logging into azure Portal",
                metadata= {
                    "logic_app_name": "",
                    "s_no": "9"
                }
            )
        ]
        return error_response

def generate_error_summary(results):
    # Generate error summary using the LLMChain
    responses = []
    for doc in results:
            response = error_summary_chain.run(document=doc.page_content)
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

    """
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
    """
    # Determine the temporary directory path
    temp_path = os.getenv('TMPDIR', '/tmp')  # Use TMPDIR environment variable if available, otherwise fallback to '/tmp'

    # Specify the token backend with the temporary path
    token_backend = FileSystemTokenBackend(token_path=temp_path, token_filename='o365_token.txt')
    # Set up the connection
    credentials = (CLIENT_ID, CLIENT_SECRET)
    account = Account(credentials, auth_flow_type='credentials', tenant_id=TENANT_ID,token_backend=token_backend)
    
    if account.authenticate():
        # Create a new message
        m = account.new_message(resource=SMTP_USERNAME)
        m.to.add(NOTIFICATION_EMAIL)
        m.subject = subject
        m.body = body
        m.body_type = 'HTML'
        # Send the message
        m.send()
        logging.info("Notification email sent successfully")
    else:
        logging.error("Failed to authenticate with Office 365")

app = func.FunctionApp()

@app.route(route="ProcessErrorSummaryEmailAPI", auth_level=func.AuthLevel.ANONYMOUS)
def ProcessErrorEmail(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    try:
        req_body = req.get_body().decode('utf-8')
        logging.info(req_body)
    except ValueError:
        return func.HttpResponse("Invalid JSON in request body", status_code=400)

    email_body = req_body
    if not email_body:
        return func.HttpResponse("No email body provided in the request", status_code=400)
    try:
        # Process the email
        text_content = extract_html_content(email_body)
        error_source = extract_error_message(text_content)
        
        # Query log layer message from Azure SQL Database
        log_layer_message = query_log_layer_message(error_source)
        
        # Query for similar errors
        similar_error = query_similar_errors(log_layer_message)
        
        # Generate error summary
        error_summary = generate_error_summary(similar_error)
        
        # Prepare notification email
        notification_subject = f"""Error Alert: New Error Processed from AI at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"""
        notification_body = f"""
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error Alert</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }}
                    .error-alert {{
                        background-color: #ffffff;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        padding: 20px;
                        max-width: 500px;
                        width: 100%;
                    }}
                    .error-header {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 15px;
                    }}
                    .error-icon {{
                        background-color: #ff4444;
                        color: white;
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin-right: 10px;
                        font-weight: bold;
                    }}
                    .error-title {{
                        font-size: 18px;
                        font-weight: bold;
                        margin: 0;
                    }}
                    .error-content {{
                        margin-bottom: 15px;
                    }}
                    .error-field {{
                        margin-bottom: 10px;
                    }}
                    .error-label {{
                        font-weight: bold;
                        margin-bottom: 5px;
                        display: block;
                    }}
                    .error-value {{
                        margin: 0;
                    }}
                    .error-recommendations {{
                        background-color: #f8f8f8;
                        border-radius: 4px;
                        padding: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="error-alert">
                    <div class="error-header">
                        <div class="error-icon">!</div>
                        <h2 class="error-title">Error Alert from Azure Failures</h2>
                    </div>
                    <div class="error-content">
                        <div class="error-field">
                            <span class="error-label">Logic app Name:</span>
                            <p class="error-value">{error_source}</p>
                        </div>
                        <div class="error-field">
                            <span class="error-label">Log Layer Message:</span>
                            <p class="error-value">{log_layer_message}</p>
                        </div>
                        <div class="error-field">
                            <span class="error-label">Error Summary:</span>
                            <p class="error-value">{error_summary["Error Summary"]}</p>
                        </div>
                    </div>
                    <div class="error-recommendations">
                        <span class="error-label">Recommendations:</span>
                        <p class="error-value">{error_summary["Recommendation"]}</p>
                    </div>
                </div>
            </body>
            </html>
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
    except Exception as execp:
        logging.error(execp)
        return func.HttpResponse(
        json.dumps({
            "message": "Error processed and notification sent", 
            "error_summary": error_summary,
            "log_layer_message": log_layer_message
        }),
        mimetype="application/json",
        status_code=400
        )
        
        
class ErrorDocumentFormat:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata