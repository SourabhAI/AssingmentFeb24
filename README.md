# IMDB Movie Query System

This system provides a RAG (Retrieval Augmented Generation) based query interface for IMDB movie data. It uses Qdrant for vector storage, Google's Gemini (or other LLMs) for generation, and supports both data vectorization and querying.

## Prerequisites

- Python 3.9+
- For Apple Silicon (M1/M2) Macs: Use Python installed via Homebrew
- Docker Engine (for running Qdrant)
  - [Install Docker on Windows](https://docs.docker.com/desktop/install/windows-install/)
  - [Install Docker on macOS](https://docs.docker.com/desktop/install/mac-install/)
  - [Install Docker on Linux](https://docs.docker.com/engine/install/)
- Google API key (for Gemini)
- OpenAI API key (optional, for GPT models)
- At least 4GB RAM

## Installation Steps

For Apple Silicon (M1/M2) Macs:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9 or later
brew install python@3.9
```

1. **Clone the repository and create virtual environment**
   ```bash
   git clone <repository-url>
   cd imdb-query-system
   python3 -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # On Windows:
   set GOOGLE_API_KEY=your_google_api_key
   # Optional for OpenAI:
   set OPENAI_API_KEY=your_openai_api_key

   # On Unix or MacOS:
   export GOOGLE_API_KEY=your_google_api_key
   # Optional for OpenAI:
   export OPENAI_API_KEY=your_openai_api_key
   ```

## Setting up Qdrant

1. **Start Qdrant using Docker**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   This will start Qdrant on localhost:6333

2. **Verify Qdrant is running**
   - Open a browser and visit: http://localhost:6333/dashboard
   - You should see the Qdrant dashboard

## Data Vectorization

1. **Prepare your data**
   - Ensure your IMDB dataset (imdb_top_1000.csv) is in the project directory
   - The CSV should contain the required columns (Series_Title, Released_Year, etc.)

2. **Run vectorization**
   Basic usage:
   ```bash
   python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv
   ```
   With rate limiting controls:
   ```bash
   # Smaller batches with longer delays
   python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv --batch_size 5 --delay 3.0
   
   # Larger batches with shorter delays (if you have higher quota)
   python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv --batch_size 20 --delay 1.0
   ```

   This will:
   - Create a new collection in Qdrant
   - Vectorize the movie data
   - Store the vectors and metadata
   - Handle rate limits automatically with exponential backoff

   Required parameters:
   - `--mode`: Operation mode (must be 'vectorize')
   - `--collection`: Name of the collection to create (e.g., 'imdb_movies')

   Optional parameters:
   - `--csv_path`: Path to the IMDB dataset CSV file (default: imdb_top_1000.csv)
   - `--batch_size`: Number of records to process in each batch (default: 10)
   - `--delay`: Delay in seconds between batches (default: 2.0)

Note: Make sure to use the same collection name when querying the data later.

## Querying the System

1. **Using Gemini (default)**
   Single query:
   ```bash
   python3 main.py --mode query --collection imdb_movies --query "What are the top 3 movies directed by Christopher Nolan?"
   ```
   
   Interactive mode:
   ```bash
   python3 main.py --mode query --collection imdb_movies
   ```

2. **Using OpenAI GPT**
   Single query:
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm openai --query "Which movies from 2019 had the highest meta scores?"
   ```
   
   Interactive mode:
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm openai
   ```

3. **Using local Llama**
   Single query:
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm llama --query "Tell me about movies starring Tom Hanks"
   ```
   
   Interactive mode:
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm llama
   ```

Required parameters:
- `--mode`: Operation mode (query)
- `--collection`: Name of the collection to query

Optional parameters:
- `--query`: The question to ask (if not provided, enters interactive mode)
- `--llm`: Choice of LLM (gemini, openai, or llama)

Note: The system uses Qdrant as the default vector store.

## Query System Features

The system provides a conversational interface with:

1. **Conversation Memory**
   - Maintains context across multiple questions
   - References previous interactions in responses
   - Provides more contextually relevant answers

2. **RAG Implementation**
   - Retrieves relevant movie information from the vector database
   - Combines retrieved context with conversation history
   - Generates accurate, contextual responses

3. **Source Attribution**
   - Every answer includes source information
   - Lists relevant movies with their metadata
   - Provides transparency in responses

## Rate Limiting and Performance

The system implements several strategies to handle API rate limits:

1. **Batch Processing**
   - Data is processed in configurable batch sizes
   - Default batch size is 10 records
   - Adjust using `--batch_size` parameter

2. **Delay Between Batches**
   - Configurable delay between processing batches
   - Default delay is 2 seconds
   - Adjust using `--delay` parameter

3. **Automatic Retry**
   - Implements exponential backoff when rate limits are hit
   - Automatically retries failed requests
   - Maximum 5 retry attempts per request

4. **Error Handling**
   - Graceful handling of rate limit errors
   - Continues processing after temporary failures
   - Detailed error reporting

If you encounter rate limit errors:
- Increase the delay between batches
- Decrease the batch size
- Or use a combination of both

## Example Queries

Once in the interactive query mode, you can try:

1. Single Questions:
   - "What are the top 3 movies directed by Christopher Nolan?"
   - "Which movies from 2019 had the highest meta scores?"
   - "Tell me about movies starring Tom Hanks with an IMDB rating above 8.5"

2. Conversational Follow-ups:
   - "Which one had the highest box office earnings?"
   - "Tell me more about its plot"
   - "How does it compare to his other movies?"
   - "Which actors appeared in multiple of these films?"

The system will maintain context between questions while providing sourced information
for each response.

Type 'exit' to quit the query mode.

## Project Structure 

- `main.py`: Entry point for running the system
- `query_engine.py`: Core logic for querying the system
- `requirements.txt`: List of dependencies
- `README.md`: This file

## Troubleshooting

1. **No Results or Empty Responses**
   - Ensure the vector database is properly initialized
   - Check if your query is clear and specific
   - Try rephrasing your question

2. **Error Messages**
   - "Document validation error": This is handled internally, but you might see fewer sources
   - "Connection error": Check if Qdrant is running
   - "API error": Verify your API keys and quotas

3. **Quality of Responses**
   - Be specific in your questions
   - Use follow-up questions for clarification
   - Check the sources provided for verification
