# Multi-Language Model Integration and Comparison

## Overview
This repository contains a Python script for integrating and comparing responses from various language models including OpenAI, Anthropic, Google Vertex AI, and PromptMule. The script demonstrates the use of these APIs to generate text based on user inputs and measure response latencies.

## Features
- Integration with multiple language models: OpenAI, Anthropic, Google Vertex AI, and PromptMule.
- Performance measurement in terms of response latencies.
- CSV logging for response comparison.
- Use of environment variables for secure API key management.

## Prerequisites
- Python 3.x
- Access to the APIs of OpenAI, Anthropic, Google Vertex AI, and PromptMule.
- An `.env` file containing API keys and other necessary configurations.

## Installation

To set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/mrodgers/multi-llm-integration.git
   cd multi-llm-integration
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_PROJECT_ID=your_google_project_id
   GOOGLE_LOCATION_ID=your_google_location_id
   PROMPTMULE_API_KEY=your_promptmule_api_key
   ```

## Usage

To run the script:

```bash
python script_name.py
```

Replace `script_name.py` with the actual name of the Python script.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to improve the functionality or address bugs.

## License

This project is licensed under [MIT License](LICENSE.md). Feel free to use it according to the license terms.

---

Developed with ❤ by Matt Rodgers
