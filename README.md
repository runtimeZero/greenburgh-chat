# Greenburgh AI Chat Assistant

A Streamlit-based chat application that provides information about the Town of Greenburgh using AI. The application combines OpenAI's GPT-4 with Pinecone vector database for accurate, context-aware responses about town regulations and information.

## Features

- 🤖 AI-powered chat interface using GPT-4
- 🔍 Vector search using Pinecone for accurate information retrieval
- 💾 Caching system for improved performance
- 📊 Performance monitoring and metrics
- 🔄 Real-time streaming responses
- 📝 Context-aware responses based on town documents

## Prerequisites

- Python 3.9+
- OpenAI API key
- Pinecone API key and index
- Virtual environment (recommended)

## Installation

1. Clone the repository:

bash
git clone <repository-url>
cd greenburgh-chat

2. Create and activate a virtual environment:

bash
For macOS/Linux
python3 -m venv venv
source venv/bin/activate


3. Install dependencies:

bash
pip install -r requirements.txt


4. Create a `.env` file in the project root with your API keys:

env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_index_name
PINECONE_NAMESPACE=prod
PINECONE_REGION=us-east-1


## Usage

1. Start the Streamlit application:
bash
streamlit run app.py


2. Open your web browser and navigate to `http://localhost:8501`

3. If the index is empty, click the "Add Test Data" button in the sidebar to populate the database with sample data

4. Start chatting! Ask questions about Greenburgh's regulations and information

## Features in Detail

### Vector Search
- Uses Pinecone for efficient similarity search
- Implements deduplication of similar content
- Caches embeddings to reduce API calls

### Performance Optimization
- Batch processing for data updates
- Response caching
- Performance monitoring metrics
- Configurable index settings

### User Interface
- Clean chat interface
- Real-time response streaming
- Performance metrics display
- Index statistics in sidebar

## Project Structure
greenburgh-chat/
├── app.py # Main application file
├── requirements.txt # Python dependencies
├── .env # Environment variables (not in repo)
├── .gitignore # Git ignore file
└── README.md # This file

## Deployment to Koyeb (Git-based)

1. Fork or push this repository to GitHub

2. Create a Koyeb account at https://app.koyeb.com/

3. In Koyeb dashboard:
   - Click "Create App"
   - Choose "GitHub" as deployment method
   - Select your repository
   - Choose the branch to deploy (usually `main`)

4. Set environment variables in Koyeb dashboard:
   - OPENAI_API_KEY
   - PINECONE_API_KEY
   - PINECONE_INDEX
   - PINECONE_NAMESPACE
   - PINECONE_REGION

5. Deploy:
   - Click "Deploy"
   - Wait for the build and deployment to complete

The app will be available at the URL provided by Koyeb after deployment.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PINECONE_INDEX` | Pinecone index name | Yes |
| `PINECONE_NAMESPACE` | Pinecone namespace | No |
| `PINECONE_REGION` | Pinecone region | Yes |