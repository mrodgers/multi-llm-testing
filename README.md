# Multi-Model Language Generation Interface

This repository contains code for an interactive language generation interface that integrates multiple language models, including OpenAI's GPT-3.5, Anthropic's Claude-2.1, and models from Google Vertex AI. The interface is built using Panel for a web-based chat experience and allows users to interact with these models in real-time.

## Features

- Integration with multiple language models including OpenAI's GPT-3.5, Anthropic's Claude-2.1, and Google Vertex AI's generative models.
- Real-time interaction through a web-based chat interface.
- Response latency measurement for each model.
- Response comparison across different models.
- CSV logging of user prompts, model responses, and response latencies.

## Setup

1. **OpenAI API Key**: Replace `'sk-ABC'` with your actual OpenAI API key.
2. **Anthropic API Key**: Replace `'sk-ant-api03-XDGXYZ'` with your actual Anthropic API key.
3. **Google Cloud Project ID**: Set `google_project_id` to your Google Cloud project ID.
4. **Google Cloud Location ID**: Set `google_location_id` to the appropriate Google Cloud location.

## Usage

Run the script to start the server:

```bash
./run.sh
```

Access the web-based interface at the provided URL to interact with the language models.

```bash
2024-01-21 20:56:53,787 Starting Bokeh server version 3.3.3 (running on Tornado 6.4)
2024-01-21 20:56:53,789 User authentication hooks NOT provided (default user enabled)
2024-01-21 20:56:53,790 Bokeh app running at: http://localhost:5006/multi-llm-testing
2024-01-21 20:56:53,790 Starting Bokeh server with process id: 33749
```

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/mrodgers/multi-llm-testing/issues) if you want to contribute.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Matt Rodgers - mrodgers.junk@gmail.com

Project Link: [https://github.com/mrodgers/multi-llm-testing.git](https://github.com/mrodgers/multi-llm-testing.git)
