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
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_project_id = os.getenv("GOOGLE_PROJECT_ID")
google_location_id = os.getenv("GOOGLE_LOCATION_ID")
promptmule_api_key = os.getenv("PROMPTMULE_API_KEY")

# Set OpenAI API key
openai.api_key = openai_api_key

# Setting up Anthropic API
anthropic = Anthropic(api_key=anthropic_api_key)

# Google Vertex AI setup
def generate_text_with_vertex_ai(query):
    # Initialize Vertex AI
    vertexai.init(project=google_project_id, location=google_location_id)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    # Query the model
    response = multimodal_model.generate_content([Part.from_text(query)])
    return response.text


# Function to write to CSV
def write_to_csv(llms, prompt, responses, latencies, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        # Write headers if the file is being created
        if not file_exists:
            headers = (
                ["Prompt"]
                + [f"Response from {llms[i]}" for i in range(len(responses))]
                + [f"Latency for {llms[i]}" for i in range(len(latencies))]
            )
            writer.writerow(headers)
        # Write the data
        writer.writerow([prompt] + responses + latencies)

def generate_text_with_anthropic(prompt):
    """
    Generates text using Anthropic's Claude 2.1.

    :param prompt: The prompt to send to the model :return: The generated text.
    """
    # Setting up Anthropic API
    anthropic = Anthropic(api_key=anthropic_api_key)
    
    # Use the API keys in your application
    # For example, setting the OpenAI API key
    try:
        # Call the OpenAI API
        response = anthropic.completions.create(
            model="claude-2.1",
            max_tokens_to_sample=512,
            temperature=0,
            prompt=f"{HUMAN_PROMPT} {prompt}\n{AI_PROMPT}"
        )
        # Return the generated text
        return response.completion

    except Exception as e:
        return f"An error occurred: {e}"

def generate_text_with_openai(prompt):
    """
    Generates text using OpenAI's GPT-3.

    :param prompt: The prompt to send to the model :return: The generated text.
    """
    # Use the API keys in your application
    # For example, setting the OpenAI API key
    openai.api_key = openai_api_key
    try:
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # You can choose other models like "davinci", "curie", etc.
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0
        )
        # Return the generated text
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"An error occurred: {e}"

# Function to get response from PromptMule API
def generate_text_with_promptmule(contents):
    # API endpoint - replace with the actual PromptMule API endpoint
    url = "https://api.promptmule.com/prompt"
    # Headers with API Key for authentication
    headers = {"x-api-key": promptmule_api_key, "Content-Type": "application/json"}
    # Data to be sent in the request
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": contents}],
        "max_tokens": "512",
        "temperature": "0",
        "api": "openai",
        "semantic": "0.99",
        "sem_num": "2"
    }
    # Make a POST request to the API
    response = requests.post(url, json=data, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"


# Function to automate testing across multiple LLMs
def test_prompts(prompts):
    print("starting prompt test...")
    response_filename = "multi_llm_auto_responses.csv"
    for contents in prompts:
        llms = []
        responses = []
        latencies = []
        print("For Prompt:", contents)
        
        # Generate response from each LLM
        for llm_name, llm_function in [
            ("OpenAI", generate_text_with_openai),
            ("Anthropic", generate_text_with_anthropic),
            ("Vertex AI", generate_text_with_vertex_ai),
            ("PromptMule", generate_text_with_promptmule),
        ]:
            print("Running LLM:", llm_name) 
            start_time = time.time()
            response = llm_function(contents)
            latency = time.time() - start_time
            print("Response from:", llm_name) 
            llms.append(llm_name)
            responses.append(response)
            latencies.append(latency)

        # Write to CSV
        write_to_csv(llms, contents, responses, latencies, response_filename)


# Example usage
#prompts = ["Tell me about Cisco Security MARS.", "What is the future of Network Security using GenAI?"]
prompts = [
    "Elaborate on the future trajectory of network security in the context of Generative AI (GenAI) advancements. Discuss how GenAI technologies are anticipated to revolutionize network security paradigms, specifically focusing on automated threat detection, predictive analytics, and adaptive security protocols. Provide insights into the integration of GenAI with existing cybersecurity frameworks, the potential challenges of implementing GenAI in legacy systems, and its implications for data privacy and ethical considerations. Additionally, speculate on the role of GenAI in countering sophisticated cyber threats, such as AI-generated phishing attacks, and the balance between automated security measures and human oversight in future network security strategies.",
    "Provide a comprehensive analysis of Cisco Security MARS (Monitoring, Analysis, and Response System), detailing its architectural design, operational mechanisms, and how it integrates with various network security protocols. Discuss its evolution, any notable upgrades or changes in its algorithmic approach over the years, and its effectiveness in real-time threat detection and response in complex network environments. Compare and contrast its functionalities with other leading network security solutions, highlighting its unique features and potential areas for improvement."
]

for i in range(10):  # This will iterate  
    print(f"Running iteration: {i}")
    test_prompts(prompts)
