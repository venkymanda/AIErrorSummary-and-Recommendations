import azure.functions as func
import logging
import json
import os
import openai
from bs4 import BeautifulSoup
import datetime
from datetime import datetime,timedelta
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
            <html>

            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Error Alert</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f4f4f4;
                    }}

                    .container {{
                        width: 100%;
                        max-width: 600px;
                        margin: 0 auto;
                        background-color: #ffffff;
                        padding: 20px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}

                    .header {{
                        text-align: center;
                        padding-bottom: 20px;
                    }}

                    .header img {{
                        max-width: 100px;
                        height: auto;
                        display: block;
                        margin: 0 auto;
                    }}

                    .alert-title {{
                        background-color: #ff0000;
                        color: #ffffff;
                        padding: 10px;
                        text-align: center;
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 20px;
                    }}

                    .content {{
                        font-size: 14px;
                        line-height: 1.6;
                    }}

                    .content table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                    }}

                    .content th,
                    .content td {{
                        border: 1px solid #dddddd;
                        text-align: left;
                        padding: 8px;
                    }}

                    .content th {{
                        background-color: #f2f2f2;
                    }}

                    .content .log-layer,
                    .content .error-summary,
                    .content .recommendations {{
                        margin-bottom: 20px;
                    }}

                    .footer {{
                        text-align: center;
                        font-size: 12px;
                        color: #777777;
                    }}
                    .error-header {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 20px;
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
                    .section-title {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #333;
                        margin-bottom: 10px;
                        border-bottom: 2px solid #007bff;
                        padding-bottom: 5px;
                    }}
                    .error-field {{
                        margin-bottom: 10px;
                    }}
                    .recommendations {{
                        margin-bottom: 20px;
                        padding: 10px;
                        border: 2px solid #007bff;
                        background-color: #e7f3ff;
                    }}
                    .error-summary {{
                        margin-bottom: 20px;
                        padding: 10px;
                        border: 2px solid #007bff;
                        background: rgb(240, 240, 240);
                    }}
                    .log-layer {{
                        margin-bottom: 20px;
                        padding: 10px;
                        border: 2px solid #007bff;
                        background: rgb(240, 240, 240);
                    }}


                    .recommendations h3 {{
                        margin-top: 0;
                    }}

                    .footer a {{
                        color: #007bff;
                        text-decoration: none;
                    }}
                </style>
            </head>

            <body>
                <div class="container">
                    <div class="header">
                        <img src="https://www.acumant.com/wp-content/uploads/2023/01/aculogo.webp" alt="Logo">
                    </div>
                    
                    <div class="error-header">
                        <div class="error-icon">!</div>
                        <h2 class="error-title">Error Alert from Azure Failures</h2>
                    </div>
                    <div class="content">
                        <table>
                            <tr>
                                <th>Name</th>
                                <td>{error_source}</td>
                            </tr>
                            <tr>
                                <th>Severity</th>
                                <td>Error</td>
                            </tr>
                            <tr>
                                <th>Resource</th>
                                <td>Logicapp</td>
                            </tr>
                            <tr>
                                <th>Search interval start time</th>
                                <td>J{(datetime.utcnow()-timedelta(minutes=15)).strftime('%b %d, %Y %H:%M:%S UTC')}</td>
                            </tr>
                            <tr>
                                <th>Search interval duration</th>
                                <td>15 min</td>
                            </tr>
                        </table>

                        <div class="log-layer">
                            <h3>Log Layer Message :</h3>
                            <p>{log_layer_message}</p>
                        </div>

                        <div class="error-summary">
                            <h3>Error Summary :</h3>
                            <p>{error_summary["Error Summary"]}</p>
                        </div>

                        <div class="recommendations">
                            <div class="section-title">Recommendations</div>
                            <p>{error_summary["Recommendation"]}</p>
                        </div>
                    </div>
                    
                    <div>
                <table style="border-spacing: 0px; border-collapse: collapse; padding: 0px; vertical-align: top; background: rgb(255, 255, 255); width: 640px; margin: 0px auto; text-align: inherit;" class="x_container x_footer-template" align="center" dir="ltr" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><td style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 20px;"><table style="border-spacing: 0px; border-collapse: collapse; padding: 0px; vertical-align: top; text-align: left; width: 100%; background-color: rgb(240, 240, 240);" align="center" class="x_wrapper x_outer-wrapper x_footer-wrapper" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><td style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 12px 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 20px;" class="x_wrapper-inner"><center style="width:100%; min-width:640px"><table style="border-spacing:0; border-collapse:collapse; padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left; padding:0; width:100%; display:table" class="x_row x_image-row" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><th style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; text-align: left; font-size: 14px; line-height: 20px; margin: 0px auto; padding: 12px 24px 0px; width: auto;" class="x_small-12 x_large-12 x_columns x_first x_last"><table style="border-spacing:0; border-collapse:collapse; padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left; width:100%" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><th style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 20px;"><table style="border-spacing:0; border-collapse:collapse; padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left; width:auto" valign="middle" class="x_social-links" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><td style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px 8px 0px 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 1;"><a style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; text-align: left; line-height: 20px; color: rgb(72, 70, 68); display: inline-block; text-decoration: underline;" data-auth="NotApplicable" rel="noopener noreferrer" target="_blank" href="https://nam.safelink.emails.azure.net/redirect/?destination=https%3A%2F%2Fwww.facebook.com%2Fmicrosoftazure&amp;p=bT1lZGRlZGIxOS1iZjU4LTRlOTgtYTVlNC1kY2E5ODlkNTRlYzMmcz00MzU5ZjJiMi1mMGQxLTQ0ZTgtOTlkMi1mYWFlZDQ1NzIyOTkmdT1hZW8mbD1mb290ZXIlM0FmYWNlYm9vaw%3D%3D" data-linkindex="3"><img style="outline:none; text-decoration:none; max-width:100%; border:none; height:16px; width:auto; clear:none; display:inline" class="x_social-facebook" title="Facebook" alt="Facebook" width="auto" height="16" src="https://images.ecomm.microsoft.com/cdn/mediahandler/azure-emails-templates/production/shared/images/templates/shared/images/logos/footer-facebook-4x.png" data-imagetype="External"></a> </td><td style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px 8px 0px 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 1;"><a style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; text-align: left; line-height: 20px; color: rgb(72, 70, 68); display: inline-block; text-decoration: underline;" data-auth="NotApplicable" rel="noopener noreferrer" target="_blank" href="https://nam.safelink.emails.azure.net/redirect/?destination=https%3A%2F%2Ftwitter.com%2Fazure&amp;p=bT1lZGRlZGIxOS1iZjU4LTRlOTgtYTVlNC1kY2E5ODlkNTRlYzMmcz00MzU5ZjJiMi1mMGQxLTQ0ZTgtOTlkMi1mYWFlZDQ1NzIyOTkmdT1hZW8mbD1mb290ZXIlM0F0d2l0dGVy" data-linkindex="4"><img style="outline:none; text-decoration:none; max-width:100%; border:none; height:16px; width:18px; clear:none; display:inline" class="x_social-twitter" title="Twitter" alt="Twitter" width="18" height="16" src="https://images.ecomm.microsoft.com/cdn/mediahandler/azure-emails-templates/production/shared/images/templates/shared/images/logos/footer-twitter-4x.png" data-imagetype="External"></a> </td><td style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px 8px 0px 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 1;"><a style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; text-align: left; line-height: 20px; color: rgb(72, 70, 68); display: inline-block; text-decoration: underline;" data-auth="NotApplicable" rel="noopener noreferrer" target="_blank" href="https://nam.safelink.emails.azure.net/redirect/?destination=https%3A%2F%2Fwww.youtube.com%2F%40MicrosoftAzure&amp;p=bT1lZGRlZGIxOS1iZjU4LTRlOTgtYTVlNC1kY2E5ODlkNTRlYzMmcz00MzU5ZjJiMi1mMGQxLTQ0ZTgtOTlkMi1mYWFlZDQ1NzIyOTkmdT1hZW8mbD1mb290ZXIlM0F5b3V0dWJl" data-linkindex="5"><img style="outline:none; text-decoration:none; max-width:100%; border:none; height:16px; width:24px; clear:none; display:inline" class="x_social-youtube" title="YouTube" alt="YouTube" width="24" height="16" src="https://images.ecomm.microsoft.com/cdn/mediahandler/azure-emails-templates/production/shared/images/templates/shared/images/logos/footer-youtube-4x.png" data-imagetype="External"></a> </td><td style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px 8px 0px 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 1;"><a style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; text-align: left; line-height: 20px; color: rgb(72, 70, 68); display: inline-block; text-decoration: underline;" data-auth="NotApplicable" rel="noopener noreferrer" target="_blank" href="https://nam.safelink.emails.azure.net/redirect/?destination=https%3A%2F%2Fwww.linkedin.com%2Fshowcase%2Fmicrosoft-developers&amp;p=bT1lZGRlZGIxOS1iZjU4LTRlOTgtYTVlNC1kY2E5ODlkNTRlYzMmcz00MzU5ZjJiMi1mMGQxLTQ0ZTgtOTlkMi1mYWFlZDQ1NzIyOTkmdT1hZW8mbD1mb290ZXIlM0FsaW5rZWRpbg%3D%3D" data-linkindex="6"><img style="outline:none; text-decoration:none; max-width:100%; border:none; height:16px; width:16px; clear:none; display:inline" class="x_social-icon" title="LinkedIn" alt="LinkedIn" width="16" height="16" src="https://images.ecomm.microsoft.com/cdn/mediahandler/azure-emails-templates/production/shared/images/templates/shared/images/logos/footer-linkedin-4x.png" data-imagetype="External"></a> </td></tr></tbody></table></th><th style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; margin: 0px; text-align: left; font-size: 14px; line-height: 20px; visibility: hidden; width: 0px; padding: 0px;" class="x_expander"></th></tr></tbody></table></th></tr></tbody></table><table style="border-spacing:0; border-collapse:collapse; padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left; padding:0; width:100%; display:table" class="x_row" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><th style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; text-align: left; font-size: 14px; line-height: 20px; margin: 0px auto; padding: 12px 24px; width: auto;" class="x_small-12 x_large-12 x_columns x_first x_last"><table style="border-spacing:0; border-collapse:collapse; padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left; width:100%" role="presentation"><tbody><tr style="padding-top:0; padding-right:0; padding-bottom:0; padding-left:0; vertical-align:top; text-align:left"><th style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; margin: 0px; text-align: left; font-size: 14px; line-height: 20px;"><p style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; margin: 0px; text-align: left; overflow-wrap: normal; font-size: 12px; line-height: 16px; color: rgb(72, 70, 68);" class="x_margin-bottom-0"><a style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; text-align: left; line-height: 20px; color: rgb(72, 70, 68); display: inline-block; text-decoration: underline;" title="Privacy Statement" data-auth="NotApplicable" rel="noopener noreferrer" target="_blank" href="https://nam.safelink.emails.azure.net/redirect/?destination=https%3A%2F%2Fgo.microsoft.com%2Ffwlink%2F%3FLinkId%3D521839&amp;p=bT1lZGRlZGIxOS1iZjU4LTRlOTgtYTVlNC1kY2E5ODlkNTRlYzMmcz00MzU5ZjJiMi1mMGQxLTQ0ZTgtOTlkMi1mYWFlZDQ1NzIyOTkmdT1hZW8mbD1wcml2YWN5LXN0YXRlbWVudA%3D%3D" data-linkindex="7">Privacy Statement</a> </p><p style="font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; padding: 0px; margin: 8px 0px 12px; text-align: left; overflow-wrap: normal; font-size: 12px; line-height: 16px; color: rgb(72, 70, 68);" class="x_margin-top-8">Acumant, <span style="display:inline-block; word-break:keep-all" class="x_no-wrap">AI Division, &ZeroWidthSpace;169 70, Solna,Sweden&ZeroWidthSpace;</span></p><img style="outline:none; text-decoration:none; max-width:100%; clear:both; display:block; height:20px; max-height:20px; width:auto" title="Microsoft" alt="Microsoft" class="x_logo-microsoft" width="auto" height="20" src="https://www.acumant.com/wp-content/uploads/2023/01/aculogo.webp" data-imagetype="External"> </th><th style="overflow-wrap: break-word; border-collapse: collapse; vertical-align: top; color: rgb(17, 16, 15); font-family: &quot;Segoe UI&quot;, SegoeUI, Roboto, &quot;Helvetica Neue&quot;, Arial, sans-serif; font-weight: 400; margin: 0px; text-align: left; font-size: 14px; line-height: 20px; visibility: hidden; width: 0px; padding: 0px;" class="x_expander"></th></tr></tbody></table></th></tr></tbody></table></center></td></tr></tbody></table></td></tr></tbody></table></div>
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