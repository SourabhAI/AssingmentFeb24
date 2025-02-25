# IMDB Movie Query System

Hey there! I've built this RAG-based query system for IMDB movie data. It uses Qdrant for vector storage, Google's Gemini (or other LLMs if you prefer) for the smarts behind it, and can handle both data vectorization and some pretty advanced query processing with smart filtering.

## Key Features

- **Enhanced Vectorization**: I've structured all movie attributes for really efficient filtering
- **Smart Query Parsing**: The system actually understands natural language and pulls out filtering criteria
- **Hybrid Search**: I've combined vector similarity with metadata filtering so you get much more accurate results
- **Multiple LLM Support**: Works with Google's Gemini, OpenAI, or local Llama models if you're concerned about privacy
- **Interactive Mode**: You get a pretty slick command line interface with some cool sample queries to play with

## Prerequisites

- Python 3.9 or newer
- If you've got an Apple Silicon Mac (M1/M2): Best to use Python installed via Homebrew
- Docker Engine (for running Qdrant)
  - [Install Docker on Windows](https://docs.docker.com/desktop/install/windows-install/)
  - [Install Docker on macOS](https://docs.docker.com/desktop/install/mac-install/)
  - [Install Docker on Linux](https://docs.docker.com/engine/install/)
- Google API key (for Gemini)
- OpenAI API key (optional, if you want to use GPT models)
- At least 4GB RAM (more is definitely better for performance)

## Installation Steps

For Apple Silicon (M1/M2) Macs:
```bash
# First get Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then grab Python 3.9 or later
brew install python@3.9
```

1. **Clone the repo and set up a virtual environment**
   ```bash
   git clone <repository-url>
   cd imdb-query-system
   python3 -m venv venv
   
   # Fire up the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API keys**
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

1. **Launch Qdrant with Docker**
   ```bash
   docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
   ```
   This gets Qdrant running on localhost:6333 and saves your data to the ./qdrant_data directory

   If you've already got a Qdrant container running on port 6333, just run:
   docker stop peaceful_kare

2. **Make sure Qdrant is actually running**
   - Open your browser and go to: http://localhost:6333/dashboard
   - You should see the Qdrant dashboard if all is well

## Data Vectorization

I've implemented a pretty cool vectorization scheme that:

1. **Organizes all movie attributes** so you can easily query them
2. **Creates smart indexes** for lightning-fast filtering
3. **Cleans up numerical data** so comparisons actually work
4. **Handles missing data** without throwing a fit

To run the vectorization:

```bash
# Basic version
python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv

# If you're hitting API limits, try this more conservative approach
python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv --batch_size 5 --delay 3.0 --verbose
```

The vectorizer processes these fields:
- Movie title
- Release year (with a bonus decade grouping)
- Certificate/Rating
- Runtime (converted to minutes for easier searching)
- Genre (both as text and as an array)
- IMDB Rating
- Meta Score
- Director
- Stars/Cast
- Vote counts
- Box office gross (normalized so you can actually filter by it)
- Plot overview

## Dealing with Rate Limits

When you're using cloud APIs, rate limits can be a pain. I've added a few tricks to handle this:

1. **Batch Processing**
   - The data gets processed in configurable chunks
   - Default is 10 records at a time
   - You can tweak this with the `--batch_size` parameter

2. **Delays Between Batches**
   - Takes a breather between batches
   - Default is 2 seconds
   - Adjust with the `--delay` parameter if needed

3. **Smart Retries**
   - Uses exponential backoff when it hits rate limits
   - Automatically retries failed requests
   - Will try up to 5 times before giving up

4. **Error Handling**
   - Doesn't just crash when rate limited
   - Keeps going after temporary hiccups
   - Gives you useful error info

If you're running into rate limit errors:
- Try increasing the delay between batches
- Or reduce the batch size
- Or just do both

Here's a more conservative setup:
```bash
python3 main.py --mode vectorize --collection imdb_movies --csv_path imdb_top_1000.csv --batch_size 5 --delay 5.0
```

## Advanced Query Processing

The cool thing about this system is how it handles complex questions:

1. **It actually reads your question** and figures out what you're asking for
2. **Builds smart database filters** based on what you want
3. **Combines semantic search with metadata** for really precise results
4. **Re-ranks everything** based on what you actually care about

Check out these sample queries:

1. **Basic factual stuff**
   - "When did The Matrix release?"
   - "Who directed Inception?"

2. **Filtering by different attributes**
   - "What are the top 5 movies of 2019 by meta score?"
   - "Top horror movies with a meta score above 85 and IMDB rating above 8"

3. **Time period searches**
   - "Top 7 comedy movies between 2010-2020 by IMDB rating"
   - "Best sci-fi movies from the 1990s"

4. **Complex analytical questions**
   - "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice"
   - "Top 10 movies with over 1M votes but lower gross earnings"

5. **Content and plot analysis**
   - "List of movies from the comedy genre where there is death or dead people involved"
   - "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies"

## Running Queries

```bash
# Interactive mode (this is the most fun)
python3 main.py --mode query --collection imdb_movies --llm gemini

# Or just ask a single question
python3 main.py --mode query --collection imdb_movies --llm gemini --query "What are the top 5 movies of 2019 by meta score?"

# If you want to see what's happening under the hood
python3 main.py --mode query --collection imdb_movies --llm gemini --verbose
```

## Using the Streamlit Chat Interface

I've also added a modern web-based chat interface using Streamlit that makes interacting with the movie database even more enjoyable.

### Running in Streamlit Mode

```bash
# Basic usage
python3 main.py --mode streamlit --collection imdb_movies --llm gemini

# If you need a different port (default is 8501)
python3 main.py --mode streamlit --collection imdb_movies --llm gemini --port 8502

# If you want to see the behind-the-scenes action
python3 main.py --mode streamlit --collection imdb_movies --llm gemini --verbose
```
# Just hit Ctrl+C in the terminal to kill the Streamlit server when you're done

### Chat Interface Features

- **Your conversation sticks around**: Chat history is saved during your session
- **Quick sample queries**: Just click the examples in the sidebar to try them
- **Pretty slick UI**: Clean design with a nice typing animation that feels natural
- **Context-aware answers**: Shows you how many results it filtered through
- **Helpful error messages**: When things go wrong, you'll know why

After you run the command, just go to http://localhost:8501 in your browser (or whatever port you specified).

### What you'll need

- Streamlit (already included in requirements.txt)
- Any modern browser
- Same prerequisites as the command-line mode

### Tips for using the chat

- Try those sample queries in the sidebar - they showcase some of the coolest features
- Your chat history sticks around during your session but disappears if you restart
- Be specific in your questions for the best results
- Hit Ctrl+C in the terminal when you're done to shut things down

## How the Query Engine Works

1. **Query Parsing**: The system reads your question and picks out:
   - Year ranges (like "between 2010 and 2020")
   - Rating thresholds (like "rating above 8.5")
   - Meta score ranges (like "meta score above 85")
   - Genres (like "horror movies", "comedy films")
   - Directors (like "directed by Spielberg")
   - Actors (like "starring Tom Hanks")
   - Vote counts (like "over 1M votes")
   - Box office numbers (like "earnings above 500M")
   - Top N requests (like "top 5 movies")

2. **Filter Building**: Turns all that info into optimized database queries

3. **Hybrid Search**:
   - Uses vector similarity for the meaning of your search
   - Applies metadata filters for exact matching
   - Re-ranks everything based on what you're actually asking for

4. **Context Building**:
   - Packages up the results into a nice format
   - Highlights the stuff you probably care about based on your question

5. **LLM Magic**:
   - Feeds everything to the language model
   - Gives you back a human-readable answer based on the actual data

## Other Options & Features

### Operation Modes

1. **Vectorize Mode**: Initial data processing and embedding
   ```bash
   python3 main.py --mode vectorize --collection imdb_movies
   ```

2. **Query Mode (CLI)**: Command-line interface for questions
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm gemini
   ```

3. **Streamlit Mode**: Web-based chat interface for the full experience
   ```bash
   python3 main.py --mode streamlit --collection imdb_movies --llm gemini
   ```

### LLM Options

1. **Google Gemini** (the default and generally fastest)
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm gemini
   ```

2. **OpenAI GPT** (if you prefer)
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm openai
   ```

3. **Local Llama** (for privacy or offline use)
   ```bash
   python3 main.py --mode query --collection imdb_movies --llm llama
   ```

### Interactive Mode Commands

When you're in interactive mode:
- Type `exit` when you want to quit
- Type `examples` to see some cool sample queries

## Troubleshooting

1. **API Key Issues**
   - Double-check your GOOGLE_API_KEY is set correctly
   - For OpenAI, make sure OPENAI_API_KEY is set if you're using it

2. **Docker Hiccups**
   - Make sure Docker is actually running before starting Qdrant
   - If something's weird, check the logs with `docker logs <container_id>`

3. **Rate Limit Headaches**
   - If you're hitting API limits, slow things down
   - Try a smaller batch size
   - Add more delay between batches
   - Check out the "Dealing with Rate Limits" section

4. **Memory Problems**
   - If it's using too much RAM, reduce the batch size
   - Make sure you've got at least 4GB of RAM available

5. **Query Weirdness**
   - If filters aren't working right, try rephrasing
   - Be more specific (like "rating above 8" instead of "good rating")

## Example Queries

Some fun queries to try:

1. **One-shot Questions**:
   - "What are the top 3 movies directed by Christopher Nolan?"
   - "Which movies from 2019 had the highest meta scores?"
   - "Tell me about movies starring Tom Hanks with an IMDB rating above 8.5"

2. **Follow-up Questions**:
   - "Which one had the highest box office earnings?"
   - "Tell me more about its plot"
   - "How does it compare to his other movies?"
   - "Which actors appeared in multiple of these films?"

Just type 'exit' when you want to quit.

## Project Structure

- `main.py`: The entry point for the whole system
- `data_vectorizer.py`: Handles all the vectorization with payload indexing
- `query_engine.py`: Where all the query parsing and filtering magic happens
- `app.py`: The Streamlit chat interface for interactive queries
- `requirements.txt`: All the dependencies you'll need
- `README.md`: This file you're reading right now
- `.gitignore`: Keeps virtual environments from being uploaded

## Making It Your Own

### Custom Vectorization Schema

If you want to tweak the vectorization in `data_vectorizer.py`:
1. Add new fields to the `_create_payload_indexes` method
2. Extend the preprocessing in `vectorize_and_upload`
3. Change the payload structure in the point creation

### Custom Query Parsing

To add new query capabilities in `query_engine.py`:
1. Add new extraction methods to the `QueryParser` class
2. Update the `parse_query` method to use your new extractors
3. Extend the `_build_filter` method to handle new filter types

## License

This project uses the MIT License - see the LICENSE file for all the legal stuff.

## Acknowledgments

- Built on the IMDB Top 1000 Movies dataset
- Powered by LangChain, Qdrant, and Google Generative AI
- Thanks to the awesome open-source AI and vectorstore communities
