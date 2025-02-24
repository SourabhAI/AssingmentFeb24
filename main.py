import argparse
import os
from data_vectorizer import IMDBDataVectorizer
from query_engine import IMDBQueryEngine



def vectorize_data(csv_path: str, collection_name: str, batch_size: int, delay: float):
    """Vectorize the IMDB dataset"""
    print("Initializing vectorization...")
    try:
        vectorizer = IMDBDataVectorizer(
            collection_name=collection_name,
            batch_size=batch_size,
            delay_between_batches=delay
        )
        
        print("Creating collection...")
        vectorizer.create_collection()
        
        print(f"Vectorizing data from {csv_path}...")
        vectorizer.vectorize_and_upload(csv_path)
        
        print("Vectorization complete!")
    except Exception as e:
        print(f"Vectorization failed: {e}")
        raise

def query_mode(collection_name: str, llm_type: str, query: str = None):
    """Run query mode with specified collection"""
    print(f"Initializing query engine with {llm_type} LLM...")
    try:
        query_engine = IMDBQueryEngine(
            collection_name=collection_name,
            llm_type=llm_type
        )

        if query:  # Single query mode
            result = query_engine.query(query)
            if result and "answer" in result:
                print("\nAnswer:", result["answer"])
            else:
                print("\nNo answer was generated. Please try rephrasing your question.")
            return

        # Interactive mode
        print("\nIMDB Movie Query System")
        print("Type 'exit' to quit")
        print("Example queries:")
        print("- What are the top 3 movies directed by Christopher Nolan?")
        print("- Which movies from 2019 had the highest meta scores?")
        print("- Tell me about movies starring Tom Hanks with an IMDB rating above 8.5")
        
        while True:
            print("\nEnter your query:")
            user_query = input("> ")
            
            if user_query.lower() == 'exit':
                break
                
            try:
                result = query_engine.query(user_query)
                if result and "answer" in result:
                    print("\nAnswer:", result["answer"])
                else:
                    print("\nNo answer was generated. Please try rephrasing your question.")
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Please try rephrasing your question or check if the vector database is properly initialized.")

    except Exception as e:
        print(f"Failed to initialize query engine: {e}")
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

    args = parser.parse_args()
    
    # Validate mode-specific requirements
    if args.mode == 'query' and not args.collection:
        parser.error('--collection is required for query mode')
    
    # Ensure GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Additional API key check for OpenAI
    if args.llm == 'openai' and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
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
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 