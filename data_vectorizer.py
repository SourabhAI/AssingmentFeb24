import os
import pandas as pd
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant

class RateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: retry_state.outcome.result()
)
def create_embedding_with_retry(embeddings, text):
    """Create embedding with retry logic"""
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        if "RATE_LIMIT_EXCEEDED" in str(e):
            print("Rate limit hit, waiting before retry...")
            raise RateLimitError(str(e))
        raise e

class IMDBDataVectorizer:
    def __init__(
        self,
        collection_name: str = "imdb_movies",
        embedding_model: str = "models/embedding-001",
        batch_size: int = 10,  # Reduced batch size
        delay_between_batches: float = 2.0  # Delay in seconds
    ):
        # Initialize embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            task_type="retrieval_query"
        )
        
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        
        # Get embedding dimension from the model
        sample_text = "Sample text for dimension check"
        sample_embedding = self.embeddings.embed_query(sample_text)
        self.embedding_dim = len(sample_embedding)

    def create_collection(self):
        """Create a new collection in Qdrant, recreating if it already exists"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            if self.collection_name in [c.name for c in collections.collections]:
                print(f"Collection '{self.collection_name}' exists. Recreating...")
                self.qdrant_client.delete_collection(self.collection_name)

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def prepare_movie_text(self, movie: Dict) -> str:
        """Prepare movie data as text for embedding"""
        return f"""
        Title: {movie['Series_Title']}
        Year: {movie['Released_Year']}
        Certificate: {movie['Certificate']}
        Runtime: {movie['Runtime']}
        Genre: {movie['Genre']}
        IMDB Rating: {movie['IMDB_Rating']}
        Overview: {movie['Overview']}
        Meta Score: {movie['Meta_score']}
        Director: {movie['Director']}
        Stars: {movie['Star1']}, {movie['Star2']}, {movie['Star3']}, {movie['Star4']}
        Votes: {movie['No_of_Votes']}
        Gross: {movie['Gross']}
        """

    def vectorize_and_upload(self, csv_path: str):
        """Vectorize IMDB data and upload to Qdrant"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['Series_Title', 'Released_Year', 'Certificate', 'Runtime', 
                              'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
                              'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        total_records = len(df)
        
        # Process in batches
        for i in range(0, total_records, self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            batch_records = batch_df.to_dict('records')
            
            # Prepare points for upload
            points = []
            for idx, movie in enumerate(batch_records, start=i):
                # Prepare text and create embedding
                movie_text = self.prepare_movie_text(movie)
                try:
                    embedding = create_embedding_with_retry(self.embeddings, movie_text)
                except Exception as e:
                    print(f"Error processing record {idx}: {e}")
                    continue

                # Create point
                point = models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "overview": movie["Overview"],  # This will be the content
                        "metadata": {  # This will be the metadata
                            "title": movie["Series_Title"],
                            "year": movie["Released_Year"],
                            "genre": movie["Genre"],
                            "director": movie["Director"],
                            "rating": movie["IMDB_Rating"],
                            "meta_score": movie["Meta_score"],
                            "gross": movie["Gross"]
                        }
                    }
                )
                points.append(point)
            
            if points:
                # Upload batch
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                print(f"Processed {min(i + self.batch_size, total_records)}/{total_records} records")
                
                # Add delay between batches
                if i + self.batch_size < total_records:
                    print(f"Waiting {self.delay_between_batches} seconds before next batch...")
                    time.sleep(self.delay_between_batches)

def main():
    # Initialize vectorizer
    vectorizer = IMDBDataVectorizer()
    
    # Create collection
    vectorizer.create_collection()
    
    # Vectorize and upload data
    vectorizer.vectorize_and_upload("imdb_top_1000.csv")

if __name__ == "__main__":
    main() 