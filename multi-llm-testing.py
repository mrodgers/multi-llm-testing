import panel as pn
from ctransformers import AutoModelForCausalLM
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import time
import csv
import os
from dotenv import load_dotenv
import requests

# Load the environment variables from .env file
load_dotenv()

# Accessing the API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

pn.extension()

# Google Vertex API, this is the google project you are using, also you will need to be logged in to google api
# pip3 install --upgrade --user google-cloud-aiplatform
# gcloud auth application-default login
google_project_id = os.getenv('GOOGLE_PROJECT_ID')
google_location_id = os.getenv('GOOGLE_LOCATION_ID')

# get PromptMule's API key set up
promptmule_api_key =os.getenv('PROMPTMULE_API_KEY')

# this is the file where responses are saved along side of each other, with latencies
response_filename = 'multi_llm_responses.csv'

# After loading the environment variables
print("OpenAI API Key:", openai_api_key)
print("Anthropic API Key:", anthropic_api_key)
print("Google Setup: ", google_location_id, google_project_id)
print("PromptMule API Key: ", promptmule_api_key)

# Model arguments for local language models
MODEL_ARGUMENTS = {
    # Define each model's arguments and keyword arguments
    "samantha": {
        "args": ["TheBloke/Dr_Samantha-7B-GGUF"],
        "kwargs": {"model_file": "dr_samantha-7b.Q5_K_M.gguf"},
    },
    "llama": {
        "args": ["TheBloke/Llama-2-7b-Chat-GGUF"],
        "kwargs": {"model_file": "llama-2-7b-chat.Q5_K_M.gguf"},
    },
    "mistral": {
        "args": ["TheBloke/Mistral-7B-Instruct-v0.1-GGUF"],
        "kwargs": {"model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"},
    },
}

# Function to generate text via Google Vertex AI
def generate_text_with_vertex_ai(project_id: str, location: str, query: str) -> str:
    if not project_id:
        print("Skipping Vertex AI due to missing Google Project ID.")
        return "Skipping Vertex AI due to missing Google Project ID."
    if not location:
        print("Skipping Vertex AI due to missing Google Project Location.")
        return "Skipping Vertex AI due to missing Google Project Location."
    
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    
    print("Sending prompt to: Vertex AI")

    # Query the model
    response = multimodal_model.generate_content(
        [
            # Add your query here
            Part.from_text(query)
        ]
    )
    return response.text

# Function to write to CSV

# Function to generate text via PromptMule's proxy cache to OpenAI
def generate_text_with_promptmule(promptmule_api_key, contents):
    if not promptmule_api_key:
        print("Skipping PromptMule due to missing API key.")
        return "PromptMule API key not provided, get one at https://app.promptmule.com"

    # API endpoint - replace with the actual PromptMule API endpoint
    url = 'https://api.promptmule.com/prompt'

    # Headers with API Key for authentication
    headers = {
        'x-api-key': promptmule_api_key,
        'Content-Type': 'application/json'
    }

    # Data to be sent in the request
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": contents  # Corrected from {contents} to contents
            }
        ],
        "max_tokens": "512",        # Corrected to integer
        "temperature": "0",         # Corrected to integer or float
        "api": "openai",
        "semantic": "0.99",         # Corrected to float
        "sem_num": "2"              # Corrected to integer
    }

    print("Sending prompt to: PromptMule")

    # Make a POST request to the API
    response = requests.post(url, json=data, headers=headers)

    response_json = response.json()
    promptmule_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Check if the request was successful
    if response.status_code == 200:
        # Return the response content
        return promptmule_response
    else:
        # Handle errors (you can expand this part based on your error handling policy)
        return f'Error: {response.status_code}, {response.text}'

# Function to generate text using Anthropic's Claude 2.1
def generate_text_with_anthropic(api_key, prompt):
    """
    Generates text using Anthropic's Claude 2.1.

    :param prompt: The prompt to send to the model :return: The generated text.
    """
    if not api_key:
        print("Skipping Anthropic due to missing API key.")
        return "Anthropic API key not provided."
    
    # Setting up Anthropic API
    print("Sending prompt to: Anthropic")
    anthropic = Anthropic(api_key=api_key)
    
    # Use the API keys in your application
    # For example, setting the Anthropic API key
    try:
        # Call the OpenAI API
        response = anthropic.completions.create(
            model="claude-2.1",
            max_tokens_to_sample=512,
            temperature=0,
            prompt=f"{HUMAN_PROMPT} {prompt}\n{AI_PROMPT}"
        )
        # Return the generated text
        return response

    except Exception as e:
        return f"An error occurred: {e}"

# Function to generate text using OpenAI's GPT-3
def generate_text_with_openai(api_key, prompt):
    """
    Generates text using OpenAI's GPT-3.

    :param prompt: The prompt to send to the model :return: The generated text.
    """
    if not api_key:
        print("Skipping OpenAI due to missing API key.")
        return "OpenAI API key not provided, get one at https://openai.com"

    # Use the API keys in your application
    # For example, setting the OpenAI API key
    openai.api_key = api_key
    print("Sending prompt to: OpenAI")

    try:
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or another model like "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512
        )
        # Return the generated text
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

# Write the gathered data to CSV
def write_to_csv(prompt, responses, latencies, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is being created
        if not file_exists:
            headers = ['Prompt'] + [f'Response from LLM {i+1}' for i in range(len(responses))] + [f'Latency for LLM {i+1}' for i in range(len(latencies))]
            writer.writerow(headers)

        # Write the data
        writer.writerow([prompt] + responses + latencies)

# Callback function for the chat interface
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    responses = []
    latencies = []
    
    # Define prompts for local LLMs
    alpaca_prompt = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {contents}

    ### Response:'''

    llama_prompt = f'''
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant.
    <</SYS>>
    {contents} [/INST]</s>'''

    mistral_prompt = f'''<s>[INST] Please respond to the Question : {contents} [/INST]'''
 
 
    # Code to interact with each API and local LLMs
    # Include checks for API keys before making calls
 
    # OpenAI response
    print("OpenAI Key: ", openai_api_key)
    openai_response = generate_text_with_openai(openai_api_key, contents)
    if openai_api_key:
        openai_latency = time.time() - start_time
        responses.append(openai_response)
        latencies.append(openai_latency)
        instance.stream(openai_response + '\nlatency: ' + str(openai_latency), user="OpenAI", message=None)
    else:
        instance.stream(openai_response, user="System", message=None)
 
    # Anthropic response
    print("Anthropic Key: ", anthropic_api_key)
    if anthropic_api_key:
        start_time = time.time()
        anthropic_response = generate_text_with_anthropic(anthropic_api_key, contents)
        anthropic_latency = time.time() - start_time
        responses.append(anthropic_response.completion)
        latencies.append(anthropic_latency)
        instance.stream(anthropic_response.completion + '\nlatency: ' + str(anthropic_latency), user="Anthropic", message=None)
    else:
        responses.append("Athropic API Key not found.")
        latencies.append(0)
        instance.stream("Anthropic API Key not found.", user="System", message=None)
 

    # Google Vertex response
    if google_project_id:
        if google_location_id:
            start_time = time.time()
            vertex_ai_response = generate_text_with_vertex_ai(google_project_id, google_location_id, contents)
            vertex_ai_latency = time.time() - start_time
            responses.append(vertex_ai_response)
            latencies.append(vertex_ai_latency)
            instance.stream(vertex_ai_response+ '\nlatency: ' + str(vertex_ai_latency), user="Google Vertex AI", message=None)
        else:
            print("Missing Vertex AI: Google Project Location")
            instance.stream("Missing Vertex AI: Google Project Location", user="System", message=None)
    else:
        print("Missing Vertex AI: Google Project ID")
        instance.stream("Missing Vertex AI: Google Project ID", user="System", message=None)

            
    # PromptMule response
    if promptmule_api_key:
        start_time = time.time()
        promptmule_response = generate_text_with_promptmule( promptmule_api_key, contents)
        promptmule_latency = time.time() - start_time
        responses.append(promptmule_response)
        latencies.append(promptmule_latency)
        instance.stream(promptmule_response + '\nlatency: ' + str(promptmule_latency), user="PromptMule", message=None)
    else:
        print("PromptMule API Key not set.")
        instance.stream("PromptMule API Key not set. Get one at https://app.promptmule.com", user="System", message=None)
 
    
    
    # delimit SaaS v. Local LLMs
    #instance.stream('Now trying local LLMs...', user="System", message=None)
 
    
    for model in MODEL_ARGUMENTS:
        if model not in pn.state.cache:
            pn.state.cache[model] = AutoModelForCausalLM.from_pretrained(
                *MODEL_ARGUMENTS[model]["args"],
                **MODEL_ARGUMENTS[model]["kwargs"],
                gpu_layers=30,
            )

        llm = pn.state.cache[model]
        
        if model == 'samantha': prompt = alpaca_prompt
        if model == 'llama': prompt = llama_prompt
        if model == 'mistral': prompt = mistral_prompt
        
        print("Sending prompt to: ", model)

        start_time = time.time()
        response = llm(prompt, max_new_tokens=512, stream=False)
        model_latency = time.time() - start_time
        
        message = None
        #for chunk in response:
        #    message = instance.stream(chunk, user=model.title(), message=message)
        instance.stream(response.strip() + '\nlatency: ' + str(model_latency), user=model.title(), message=message)
        
        responses.append(response.strip())
        latencies.append(model_latency)
        
    # Write to CSV
    write_to_csv(contents, responses, latencies, response_filename)

# Initialize the chat interface and set it up for serving
chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send(
    "Send a message to get a response from each: PromptMule, Llama 2, Mistral (7B), Google, Anthropic and OpenAI!",
    user="System",
    respond=False,
)
chat_interface.servable()
