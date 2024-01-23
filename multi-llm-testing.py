import panel as pn
from ctransformers import AutoModelForCausalLM
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
#from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import time
import csv
import asyncio
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Accessing the API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

# Use the API keys in your application
# For example, setting the OpenAI API key
openai.api_key = openai_api_key

# Setting up Anthropic API
anthropic = Anthropic(api_key=anthropic_api_key)

pn.extension()

# Google Vertex API, this is the google project you are using, also you will need to be logged in to google api
# pip3 install --upgrade --user google-cloud-aiplatform
# gcloud auth application-default login
google_project_id = os.getenv('GOOGLE_PROJECT_ID')
google_location_id = os.getenv('GOOGLE_LOCATION_ID')

# this is the file where responses are saved along side of each other, with latencies
response_filename = 'llm_responses.csv'

# After loading the environment variables
print("OpenAI API Key:", openai_api_key)
print("Anthropic API Key:", anthropic_api_key)
print("Google Setup: ", google_location_id, google_project_id)

MODEL_ARGUMENTS = {
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

# Google Vertex AI setup
def generate_text_with_vertex_ai(project_id: str, location: str, query: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    # Query the model
    response = multimodal_model.generate_content(
        [
            # Add your query here
            Part.from_text(query)
        ]
    )
    return response.text

# Function to write to CSV
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

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    responses = []
    latencies = []
    
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

    
    # OpenAI response
    start_time = time.time()

    openai_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or another model like "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": contents}
        ],
        max_tokens=512
    )
    openai_latency = time.time() - start_time
    responses.append(openai_response.choices[0].message.content)
    latencies.append(openai_latency)
    instance.stream(openai_response.choices[0].message.content + '\nlatency: ' + str(openai_latency), user="OpenAI", message=None)
    
    # Anthropic response
    start_time = time.time()
    anthropic_response = anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=512,
        temperature=0,
        prompt=f"{HUMAN_PROMPT} {contents}\n{AI_PROMPT}"
    )
    anthropic_latency = time.time() - start_time
    responses.append(anthropic_response.completion)
    latencies.append(anthropic_latency)
    instance.stream(anthropic_response.completion + '\nlatency: ' + str(anthropic_latency), user="Anthropic", message=None)

    # Google Vertex response
    start_time = time.time()
    vertex_ai_response = generate_text_with_vertex_ai(google_project_id, google_location_id, contents)
    vertex_ai_latency = time.time() - start_time
    responses.append(vertex_ai_response)
    latencies.append(vertex_ai_latency)
    instance.stream(vertex_ai_response+ '\nlatency: ' + str(vertex_ai_latency), user="Google Vertex AI", message=None)

    
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


chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send(
    "Send a message to get a response from each: Llama 2, Mistral (7B), Google, Anthropic and OpenAI!",
    user="System",
    respond=False,
)
chat_interface.servable()
