# Multi-Model Language Generation Interface

This repository contains code for an interactive language generation interface that integrates multiple language models, including OpenAI's GPT-3.5, Anthropic's Claude-2.1, and models from Google Vertex AI. The interface is built using Panel for a web-based chat experience and allows users to interact with these models in real-time.

## Features

- Integration with multiple language models including OpenAI's GPT-3.5, Anthropic's Claude-2.1, and Google Vertex AI's generative models.
- Real-time interaction through a web-based chat interface.
- Response latency measurement for each model.
- Response comparison across different models.
- CSV logging of user prompts, model responses, and response latencies.

## Installation

Before running the script, ensure you have the required dependencies installed:

```bash
pip install panel openai google-cloud-aiplatform
```

## Setup

1. **OpenAI API Key**: Replace `'sk-ABC'` with your actual OpenAI API key.
2. **Anthropic API Key**: Replace `'sk-ant-api03-XDGXYZ'` with your actual Anthropic API key.
3. **Google Cloud Project ID**: Set `google_project_id` to your Google Cloud project ID.
4. **Google Cloud Location ID**: Set `google_location_id` to the appropriate Google Cloud location.

## Usage

Run the script to start the server:

```bash
python your_script_name.py
```

Access the web-based interface at the provided URL to interact with the language models.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page]() if you want to contribute.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Matt Rodgers - mrodgers.junk@gmail.com

Project Link: [https://github.com/](https://github.com/)