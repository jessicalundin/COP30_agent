# Climate & Health RAG Assistant

A Streamlit application that uses Retrieval-Augmented Generation (RAG) to answer questions about climate change, COP30, Brazil, and Dengue fever, while also providing data visualizations and carbon emissions tracking.

## Features

- **RAG Question-Answering System**: Ask questions about climate change, COP30, Brazil, and Dengue fever to get informed responses backed by documents.
- **Multiple LLM Options**: Choose between OpenAI GPT-3.5 or local Llama 3.2 models (1B or 3B) for inference.
- **User Authentication**: Protect your application and API key with username and password authentication.
- **Carbon Emissions Tracking**: See the carbon footprint of your queries in real-time and compare different models.
- **Document Upload**: Upload your own PDF documents to expand the knowledge base.
- **Global Climate Data Visualization**: Explore global temperature trends with interactive charts and graphs.
- **Brazil & Dengue Analysis**: Visualize the relationship between climate variables and Dengue fever cases in Brazil.
- **Multiple Visualization Options**: Choose between line charts, bar charts, and interactive plots.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/COP30_agent.git
cd COP30_agent
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Set up your environment variables:
Create a `.env` file in the root directory with your configuration:
```
# OpenAI API Key (if using OpenAI models)
OPENAI_API_KEY=your_api_key_here

# Authentication credentials (change these for security)
APP_USERNAME=
APP_PASSWORD=

# Hugging Face Token (HF_TOKEN)
HF_TOKEN=your_token_here
```

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```

2. Open your web browser and navigate to the URL provided in the terminal (typically http://localhost:8501).

3. Login with the credentials you set in the `.env` file (defaults to username: `your username`, password: `your password`).

4. In the sidebar:
   - Select your preferred model (OpenAI GPT-3.5 or Llama 3.2)
   - Enter your OpenAI API key if using GPT-3.5 and not provided during login
   - Toggle carbon emissions tracking

5. Navigate between tabs to access different features:
   - **RAG Assistant**: Ask questions and get answers with carbon footprint tracking
   - **Climate Data**: Explore global temperature trends
   - **Brazil & Dengue**: Visualize the relationship between climate change and Dengue fever in Brazil

## Authentication System

The application includes a basic authentication system to protect your API key and restrict access to authorized users:

- Default credentials: username `your username`, password `your password`
- You can change these by setting `APP_USERNAME` and `APP_PASSWORD` in your `.env` file
- Users can provide an OpenAI API key during login or later in the sidebar
- The authentication uses session management, so users will need to log in again if they close their browser

## Carbon Emissions Tracking

This application includes real-time carbon emissions tracking for LLM inference:

- **OpenAI Models**: Emissions are estimated based on tokens processed and research on cloud API carbon intensity
- **Llama Models**: Emissions are measured in real-time using the CodeCarbon library, which monitors your device's power consumption

The application shows:
- Carbon footprint per query in mg CO₂e
- Comparisons to everyday activities
- Cumulative emissions for your session
- Comparative emissions between different model types

## Data Sources

- Global temperature data: NASA GISS Surface Temperature Analysis
- Brazil climate and Dengue data: Simulated data based on trends from scientific literature and reports
- COP30 and Dengue information: Compiled from authoritative sources
- Carbon emissions data: CodeCarbon measurements and published research

## Requirements

- Python 3.8+
- See requirements.txt for all Python package dependencies
- For Llama models: 8GB+ RAM recommended, GPU acceleration optional but recommended

## Note

When using Llama models locally, initial loading may take some time depending on your hardware. The models are optimized with 4-bit quantization to reduce memory requirements.

## License

MIT 

## Environment Setup

### Required Environment Variables

Create a `.env` file in the root directory with the following variables: 