import argparse
import os
import logging
import sys
import warnings

# Let's silence those annoying warnings
warnings.filterwarnings('ignore')
# Specifically muzzle those LangChain deprecation warnings - they're super chatty
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')

from data_vectorizer import IMDBDataVectorizer
from query_engine import IMDBQueryEngine

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('imdb_query_system')

def setup_environment():
    """Make sure we've got all the API keys we need"""
    required_vars = ['GOOGLE_API_KEY']
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    # Check for OpenAI API key too, but only if they want to use OpenAI
    if len(sys.argv) > 1 and '--llm' in sys.argv and 'openai' in sys.argv[sys.argv.index('--llm') + 1]:
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY environment variable not set, but OpenAI LLM requested")
            print("\nWARNING: You asked for OpenAI but I don't see an API key set up.")
            confirm = input("Want to continue anyway? (y/n): ")
            if confirm.lower() != 'y':
                sys.exit(1)

def vectorize_data(csv_path: str, collection_name: str, batch_size: int, delay: float):
    """Turn the IMDB dataset into vectors with our enhanced schema"""
    logger.info("Starting up the vectorization process...")
    try:
        vectorizer = IMDBDataVectorizer(
            collection_name=collection_name,
            batch_size=batch_size,
            delay_between_batches=delay
        )
        
        logger.info("Creating a fresh collection with all the indexes we need...")
        vectorizer.create_collection()
        
        logger.info(f"Let's start vectorizing the data from {csv_path}...")
        vectorizer.vectorize_and_upload(csv_path)
        
        logger.info("All done with vectorization! Data is ready to query.")
    except Exception as e:
        logger.error(f"Shoot, vectorization failed: {e}", exc_info=True)
        raise

def display_sample_queries():
    """Show off some cool queries people can try"""
    sample_queries = [
        "When did The Matrix release?",
        "What are the top 5 movies of 2019 by meta score?",
        "Top 7 comedy movies between 2010-2020 by imdb rating?",
        "Top horror movies with a meta score above 85 and imdb rating above 8",
        "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice.",
        "Top 10 movies with over 1M votes but lower gross earnings.",
        "List of movies from the comedy genre where there is death or dead people involved.",
        "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies."
    ]
    
    print("\nHere are some fun queries to try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    print()

def query_mode(collection_name: str, llm_type: str, query: str = None):
    """Fire up the query engine with all the bells and whistles"""
    logger.info(f"Starting the query engine with {llm_type}...")
    try:
        query_engine = IMDBQueryEngine(
            collection_name=collection_name,
            llm_type=llm_type
        )

        if query:  # They just want to ask one question
            logger.info(f"Answering: {query}")
            result = query_engine.query(query)
            if result and "answer" in result:
                print("\nAnswer:", result["answer"])
                
                # Tell them if we used filtering magic
                if result.get("filtered_search"):
                    print(f"\n(Found from {result.get('num_results', 'several')} filtered results)")
            else:
                print("\nHmm, I couldn't figure that one out. Maybe try rephrasing?")
            return

        # Interactive mode - much more fun
        print("\nIMDB Movie Query System")
        print("Type 'exit' when you want to quit")
        display_sample_queries()
        
        while True:
            print("\nWhat do you want to know?")
            user_query = input("> ")
            
            if user_query.lower() == 'exit':
                break
                
            if user_query.lower() == 'examples':
                display_sample_queries()
                continue
                
            try:
                logger.info(f"Looking up: {user_query}")
                result = query_engine.query(user_query)
                if result and "answer" in result:
                    print("\nAnswer:", result["answer"])
                    
                    # Show extra info about filtered results
                    if result.get("filtered_search"):
                        print(f"\n(Found from {result.get('num_results', 'several')} filtered results)")
                else:
                    print("\nHmm, I couldn't figure that one out. Maybe try rephrasing?")
            except Exception as e:
                logger.error(f"Oops, hit a snag: {e}", exc_info=True)
                print(f"\nSorry, something went wrong: {str(e)}")
                print("Try rephrasing your question or check if the database is properly set up.")

    except Exception as e:
        logger.error(f"Couldn't start the query engine: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='IMDB Dataset Vectorization and Query System')
    
    parser.add_argument(
        '--mode',
        choices=['vectorize', 'query', 'streamlit'],
        required=True,
        help='What do you want to do: vectorize data, query the database, or launch the streamlit interface'
    )
    
    # Common stuff everyone needs
    parser.add_argument(
        '--collection',
        help='Name of the collection to use (required for all modes)',
        required=True
    )
    
    # Vectorization settings
    parser.add_argument(
        '--csv_path',
        default='imdb_top_1000.csv',
        help='Path to the IMDB CSV file (for vectorize mode)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='How many records to process at once (for vectorize mode)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='How many seconds to wait between batches (for vectorize mode)'
    )
    
    # Query settings
    parser.add_argument(
        '--llm',
        choices=['gemini', 'openai', 'llama'],
        default='gemini',
        help='Which AI model to use for answering (for query mode)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Direct question to ask (optional for query mode). If not provided, starts interactive mode'
    )
    
    # Streamlit settings
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Which port to run Streamlit on (for streamlit mode)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show more details about what\'s happening under the hood'
    )

    args = parser.parse_args()
    
    # Crank up the logging if they want to see all the details
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Extra verbose mode turned on - you'll see everything")
    
    # Make sure they provided all the right args
    if args.mode in ('query', 'streamlit') and not args.collection:
        parser.error('You need to specify --collection for query and streamlit modes')
    
    # Check for API keys
    try:
        setup_environment()
    except EnvironmentError as e:
        logger.error(str(e))
        print(f"Error: {e}")
        print("Please set up your API keys first.")
        exit(1)
    
    try:
        if args.mode == 'vectorize':
            vectorize_data(
                csv_path=args.csv_path,
                collection_name=args.collection,
                batch_size=args.batch_size,
                delay=args.delay
            )
        elif args.mode == 'streamlit':
            print(f"Starting the Streamlit interface for '{args.collection}' with {args.llm}...")
            run_streamlit(
                collection_name=args.collection,
                llm_type=args.llm,
                port=args.port
            )
        else:  # query mode
            query_mode(
                collection_name=args.collection,
                llm_type=args.llm,
                query=args.query
            )
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}")
        exit(1)

def run_streamlit(collection_name: str, llm_type: str, port: int = 8501):
    """Fire up the Streamlit chat interface"""
    try:
        import streamlit.web.bootstrap as bootstrap
        import app
        
        # Initialize the query engine first so it's ready when the app loads
        query_engine = IMDBQueryEngine(
            collection_name=collection_name,
            llm_type=llm_type
        )
        
        # Pass config to the app via environment vars
        import os
        os.environ['IMDB_COLLECTION_NAME'] = collection_name
        os.environ['IMDB_LLM_TYPE'] = llm_type
        
        # Start the Streamlit server
        logger.info(f"Starting Streamlit on port {port}...")
        print(f"\nLaunching the chat UI at http://localhost:{port}")
        print("Hit Ctrl+C when you're done to shut it down")
        
        # Light it up!
        bootstrap.run(
            app.__file__,
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        )
    except ImportError:
        logger.error("Oops, looks like Streamlit isn't installed. You need to install it first.")
        print("\nError: Streamlit isn't installed.")
        print("Run this first: pip install streamlit")
        exit(1)
    except Exception as e:
        logger.error(f"Problem starting Streamlit: {e}", exc_info=True)
        print(f"Error launching Streamlit: {e}")
        exit(1)

if __name__ == "__main__":
    main() 