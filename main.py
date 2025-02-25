import argparse
import os
import logging
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# Specifically suppress LangChain deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')

from data_vectorizer import IMDBDataVectorizer
from query_engine import IMDBQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('imdb_query_system')

def setup_environment():
    """Verify required environment variables are set"""
    required_vars = ['GOOGLE_API_KEY']
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    # Optional check for OpenAI API key if using OpenAI model
    if len(sys.argv) > 1 and '--llm' in sys.argv and 'openai' in sys.argv[sys.argv.index('--llm') + 1]:
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY environment variable not set, but OpenAI LLM requested")
            print("\nWARNING: OPENAI_API_KEY environment variable not set, but OpenAI LLM requested.")
            confirm = input("Continue anyway? (y/n): ")
            if confirm.lower() != 'y':
                sys.exit(1)

def vectorize_data(csv_path: str, collection_name: str, batch_size: int, delay: float):
    """Vectorize the IMDB dataset with enhanced schema"""
    logger.info("Initializing vectorization...")
    try:
        vectorizer = IMDBDataVectorizer(
            collection_name=collection_name,
            batch_size=batch_size,
            delay_between_batches=delay
        )
        
        logger.info("Creating collection with payload indexes...")
        vectorizer.create_collection()
        
        logger.info(f"Vectorizing data from {csv_path}...")
        vectorizer.vectorize_and_upload(csv_path)
        
        logger.info("Vectorization complete!")
    except Exception as e:
        logger.error(f"Vectorization failed: {e}", exc_info=True)
        raise

def display_sample_queries():
    """Display sample queries that showcase the system's capabilities"""
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
    
    print("\nSample queries to try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    print()

def query_mode(collection_name: str, llm_type: str, query: str = None):
    """Run query mode with advanced filtering and retrieval"""
    logger.info(f"Initializing query engine with {llm_type} LLM...")
    try:
        query_engine = IMDBQueryEngine(
            collection_name=collection_name,
            llm_type=llm_type
        )

        if query:  # Single query mode
            logger.info(f"Processing query: {query}")
            result = query_engine.query(query)
            if result and "answer" in result:
                print("\nAnswer:", result["answer"])
                
                # If filtered search was used, show that information
                if result.get("filtered_search"):
                    print(f"\n(Retrieved from {result.get('num_results', 'unknown')} filtered results)")
            else:
                print("\nNo answer was generated. Please try rephrasing your question.")
            return

        # Interactive mode
        print("\nIMDB Movie Query System")
        print("Type 'exit' to quit")
        display_sample_queries()
        
        while True:
            print("\nEnter your query:")
            user_query = input("> ")
            
            if user_query.lower() == 'exit':
                break
                
            if user_query.lower() == 'examples':
                display_sample_queries()
                continue
                
            try:
                logger.info(f"Processing interactive query: {user_query}")
                result = query_engine.query(user_query)
                if result and "answer" in result:
                    print("\nAnswer:", result["answer"])
                    
                    # If filtered search was used, show that information
                    if result.get("filtered_search"):
                        print(f"\n(Retrieved from {result.get('num_results', 'unknown')} filtered results)")
                else:
                    print("\nNo answer was generated. Please try rephrasing your question.")
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nError processing query: {str(e)}")
                print("Please try rephrasing your question or check if the vector database is properly initialized.")

    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='IMDB Dataset Vectorization and Query System')
    
    parser.add_argument(
        '--mode',
        choices=['vectorize', 'query'],
        required=True,
        help='Operation mode: vectorize data or query existing database'
    )
    
    # Common arguments
    parser.add_argument(
        '--collection',
        help='Name of the collection to use (required for both modes)',
        required=True
    )
    
    # Vectorization mode arguments
    parser.add_argument(
        '--csv_path',
        default='imdb_top_1000.csv',
        help='Path to the IMDB CSV file (required for vectorize mode)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Number of records to process in each batch (vectorize mode only)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay in seconds between batches (vectorize mode only)'
    )
    
    # Query mode arguments
    parser.add_argument(
        '--llm',
        choices=['gemini', 'openai', 'llama'],
        default='gemini',
        help='LLM to use for querying (query mode only)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Direct query to run (optional for query mode). If not provided, enters interactive mode'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate mode-specific requirements
    if args.mode == 'query' and not args.collection:
        parser.error('--collection is required for query mode')
    
    # Check environment
    try:
        setup_environment()
    except EnvironmentError as e:
        logger.error(str(e))
        print(f"Error: {e}")
        print("Please set the required environment variables and try again.")
        exit(1)
    
    try:
        if args.mode == 'vectorize':
            vectorize_data(
                csv_path=args.csv_path,
                collection_name=args.collection,
                batch_size=args.batch_size,
                delay=args.delay
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

if __name__ == "__main__":
    main() 