# IMDB Movie Query System

This system provides a RAG (Retrieval Augmented Generation) based query interface for IMDB movie data. It uses Qdrant for vector storage, Google's Gemini (or other LLMs) for generation, and supports both data vectorization and advanced query processing with intelligent filtering.

## Key Features

- **Enhanced Vectorization**: All movie attributes are properly structured and indexed for efficient filtering
- **Intelligent Query Parsing**: Automatically extracts filtering criteria from natural language queries
- **Hybrid Search**: Combines vector similarity with metadata filtering for precise results
- **Multiple LLM Support**: Works with Google's Gemini, OpenAI, or local Llama models
- **Interactive Mode**: User-friendly command line interface with sample queries

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
   docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
   ```
   This will start Qdrant on localhost:6333 and persist data to the ./qdrant_data directory

   Run the following if Qdrant contatiner is already running on port 6333
   docker stop peaceful_kare

2. **Verify Qdrant is running**
   - Open a browser and visit: http://localhost:6333/dashboard
   - You should see the Qdrant dashboard

## Data Vectorization

The system implements an enhanced vectorization scheme that:

1. **Structures all movie attributes** in a queryable format
2. **Creates payload indexes** for efficient filtering
3. **Preprocesses data** for numerical comparisons
4. **Handles missing data** gracefully

To run the vectorization process:

```bash
# Basic usage
python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv

# With rate limiting controls
python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv --batch_size 5 --delay 3.0 --verbose
```

The vectorizer processes the following fields:
- Movie title
- Release year (with decade grouping)
- Certificate/Rating
- Runtime (converted to minutes)
- Genre (as both text and array)
- IMDB Rating
- Meta Score
- Director
- Stars/Cast
- Vote counts
- Box office gross (normalized to numeric values)
- Plot overview

## Rate Limiting and Performance

The system implements several strategies to handle API rate limits, which is particularly important when using cloud-based embedding services:

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

Example for more conservative rate limiting:
```bash
python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv --batch_size 5 --delay 5.0
```

## Advanced Query Processing

The system can handle complex natural language queries by:

1. **Parsing filtering criteria** from natural language
2. **Building optimized filters** for the vector database
3. **Combining vector similarity with metadata filtering**
4. **Re-ranking results** based on query intent

Example queries that demonstrate these capabilities:

1. **Simple factual questions**
   - "When did The Matrix release?"
   - "Who directed Inception?"

2. **Filtering by attributes**
   - "What are the top 5 movies of 2019 by meta score?"
   - "Top horror movies with a meta score above 85 and IMDB rating above 8"

3. **Time period queries**
   - "Top 7 comedy movies between 2010-2020 by IMDB rating"
   - "Best sci-fi movies from the 1990s"

4. **Complex analytical queries**
   - "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice"
   - "Top 10 movies with over 1M votes but lower gross earnings"

5. **Content analysis questions**
   - "List of movies from the comedy genre where there is death or dead people involved"
   - "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies"

## Running Queries

```bash
# Interactive mode (recommended)
python3 main.py --mode query --collection imdb_movies --llm gemini

# Single query mode
python3 main.py --mode query --collection imdb_movies --llm gemini --query "What are the top 5 movies of 2019 by meta score?"

# With verbose logging
python3 main.py --mode query --collection imdb_movies --llm gemini --verbose
```

## How the Query Engine Works

1. **Query Parsing**: The system analyzes natural language queries to extract:
   - Year ranges (e.g., "between 2010 and 2020")
   - Rating thresholds (e.g., "rating above 8.5")
   - Meta score ranges (e.g., "meta score above 85")
   - Genres (e.g., "horror movies", "comedy films")
   - Directors (e.g., "directed by Spielberg")
   - Actors (e.g., "starring Tom Hanks")
   - Vote thresholds (e.g., "over 1M votes")
   - Gross earnings thresholds (e.g., "earnings above 500M")
   - Top N requests (e.g., "top 5 movies")

2. **Filter Construction**: Converts parsed criteria into optimized Qdrant filters

3. **Hybrid Search**:
   - Uses vector similarity for semantic relevance
   - Applies metadata filters for precise attribute matching
   - Re-ranks results based on query intent

4. **Context Preparation**:
   - Formats filtered results into structured context
   - Highlights relevant attributes based on query type

5. **LLM Integration**:
   - Sends prepared context to the chosen LLM
   - Generates comprehensive answers based on filtered data

## Additional Options and Features

### Supported LLMs

1. **Google Gemini** (default)
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm gemini
   ```

2. **OpenAI GPT**
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm openai
   ```

3. **Local Llama**
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm llama
   ```

### Interactive Mode Commands

When in interactive mode:
- Type `exit` to quit
- Type `examples` to show sample queries

## Troubleshooting

1. **Environment Variables**
   - Ensure GOOGLE_API_KEY is set
   - For OpenAI, ensure OPENAI_API_KEY is set

2. **Docker Issues**
   - Ensure Docker is running before starting Qdrant
   - Check Qdrant logs with `docker logs <container_id>`

3. **Rate Limiting**
   - If hitting API rate limits, increase the delay between batches
   - For vectorization, use smaller batch sizes (e.g., --batch_size 5)
   - See the "Rate Limiting and Performance" section for more details

4. **Memory Issues**
   - Reduce batch size if running out of memory
   - Ensure at least 4GB of RAM is available

5. **Query Problems**
   - If filters aren't being detected, try rephrasing the query
   - Use more specific language for filters (e.g., "rating above 8" instead of "good rating")

## Example Queries

Once in the interactive query mode, you can try:

1. **Single Questions**:
   - "What are the top 3 movies directed by Christopher Nolan?"
   - "Which movies from 2019 had the highest meta scores?"
   - "Tell me about movies starring Tom Hanks with an IMDB rating above 8.5"

2. **Conversational Follow-ups**:
   - "Which one had the highest box office earnings?"
   - "Tell me more about its plot"
   - "How does it compare to his other movies?"
   - "Which actors appeared in multiple of these films?"

Type 'exit' to quit the query mode.

## Project Structure

- `main.py`: Entry point for running the system
- `data_vectorizer.py`: Enhanced vectorization with payload indexing
- `query_engine.py`: Advanced query parsing and filtering
- `requirements.txt`: List of dependencies
- `README.md`: Documentation (this file)
- `.gitignore`: Prevents virtual environments from being uploaded

## Advanced Customization

### Custom Vectorization Schema

You can modify the vectorization schema in `data_vectorizer.py` by:
1. Adding new fields to the `_create_payload_indexes` method
2. Extending the preprocessing in `vectorize_and_upload`
3. Modifying the payload structure in the point creation

### Custom Query Parsing

To extend the query parsing capabilities in `query_engine.py`:
1. Add new extraction methods to the `QueryParser` class
2. Update the `parse_query` method to use new extractors
3. Extend the `_build_filter` method to support new filter types

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project utilizes the IMDB Top 1000 Movies dataset
- Built with LangChain, Qdrant, and Google Generative AI
- Special thanks to the open-source AI and vectorstore communities
