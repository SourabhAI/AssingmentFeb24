import os
import pandas as pd
import time
import re
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue, PayloadSchemaType
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
        """Create a new collection in Qdrant with advanced payload schema for filtering"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            if self.collection_name in [c.name for c in collections.collections]:
                print(f"Collection '{self.collection_name}' exists. Recreating...")
                self.qdrant_client.delete_collection(self.collection_name)

            # Create collection with vector params
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
                # Define payload schema for efficient filtering
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0  # Index all vectors immediately
                )
            )
            
            # Create payload indexes for efficient filtering
            self._create_payload_indexes()
            
            print(f"Collection '{self.collection_name}' created successfully with payload indexes")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        # Index fields for filtering
        indexes = [
            ("year", "metadata.year", "integer"),
            ("rating", "metadata.rating", "float"),
            ("meta_score", "metadata.meta_score", "integer"),
            ("votes", "metadata.votes", "integer"),
            ("gross_numeric", "metadata.gross_numeric", "float"),
            ("genres", "metadata.genres", "keyword"),
            ("director", "metadata.director", "keyword"),
            ("stars", "metadata.stars", "keyword"),
            ("certificate", "metadata.certificate", "keyword"),
            ("runtime_minutes", "metadata.runtime_minutes", "integer"),
            ("decade", "metadata.decade", "integer")
        ]
        
        for name, path, field_type in indexes:
            schema_type = None
            if field_type == "integer":
                schema_type = PayloadSchemaType.INTEGER
            elif field_type == "float":
                schema_type = PayloadSchemaType.FLOAT
            elif field_type == "keyword":
                schema_type = PayloadSchemaType.KEYWORD
            
            if schema_type:
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=path,
                        field_schema=schema_type
                    )
                    print(f"Created index for {path}")
                except Exception as e:
                    print(f"Error creating index for {path}: {e}")

    def _parse_runtime(self, runtime_str: str) -> int:
        """Parse runtime string (e.g., '142 min') to minutes (integer)"""
        if pd.isna(runtime_str) or not runtime_str:
            return 0
        
        match = re.search(r'(\d+)', str(runtime_str))
        if match:
            return int(match.group(1))
        return 0

    def _parse_gross(self, gross_str: Union[str, float, int]) -> float:
        """Parse gross string to numeric value"""
        if pd.isna(gross_str) or not gross_str:
            return 0.0
        
        if isinstance(gross_str, (int, float)):
            return float(gross_str)
            
        # Remove non-numeric characters except decimal point
        gross_str = str(gross_str)
        gross_str = re.sub(r'[^\d.]', '', gross_str)
        
        try:
            return float(gross_str) if gross_str else 0.0
        except ValueError:
            return 0.0

    def _get_decade(self, year: Union[str, int]) -> int:
        """Get decade from year (e.g., 1995 -> 1990)"""
        if pd.isna(year) or not year:
            return 0
            
        try:
            year_num = int(year)
            return (year_num // 10) * 10
        except (ValueError, TypeError):
            return 0

    def _split_genres(self, genre_str: str) -> List[str]:
        """Split genre string into list of individual genres"""
        if pd.isna(genre_str) or not genre_str:
            return []
            
        # Split by comma and remove whitespace
        genres = [g.strip() for g in str(genre_str).split(',')]
        return [g for g in genres if g]  # Filter out empty strings

    def _split_stars(self, movie: Dict) -> List[str]:
        """Extract all stars from movie data"""
        stars = []
        for star_key in ['Star1', 'Star2', 'Star3', 'Star4']:
            if star_key in movie and movie[star_key] and not pd.isna(movie[star_key]):
                stars.append(str(movie[star_key]).strip())
        return stars

    def prepare_movie_text(self, movie: Dict) -> str:
        """Prepare movie data as comprehensive text for embedding"""
        # Ensure non-null values
        year = str(movie.get('Released_Year', '')) if not pd.isna(movie.get('Released_Year', '')) else ''
        certificate = str(movie.get('Certificate', '')) if not pd.isna(movie.get('Certificate', '')) else ''
        runtime = str(movie.get('Runtime', '')) if not pd.isna(movie.get('Runtime', '')) else ''
        genre = str(movie.get('Genre', '')) if not pd.isna(movie.get('Genre', '')) else ''
        rating = str(movie.get('IMDB_Rating', '')) if not pd.isna(movie.get('IMDB_Rating', '')) else ''
        overview = str(movie.get('Overview', '')) if not pd.isna(movie.get('Overview', '')) else ''
        meta_score = str(movie.get('Meta_score', '')) if not pd.isna(movie.get('Meta_score', '')) else ''
        director = str(movie.get('Director', '')) if not pd.isna(movie.get('Director', '')) else ''
        star1 = str(movie.get('Star1', '')) if not pd.isna(movie.get('Star1', '')) else ''
        star2 = str(movie.get('Star2', '')) if not pd.isna(movie.get('Star2', '')) else ''
        star3 = str(movie.get('Star3', '')) if not pd.isna(movie.get('Star3', '')) else ''
        star4 = str(movie.get('Star4', '')) if not pd.isna(movie.get('Star4', '')) else ''
        votes = str(movie.get('No_of_Votes', '')) if not pd.isna(movie.get('No_of_Votes', '')) else ''
        gross = str(movie.get('Gross', '')) if not pd.isna(movie.get('Gross', '')) else ''
        
        # Create rich text representation with labeled sections for better embedding
        return f"""
        Movie: {movie.get('Series_Title', '')}
        Released: {year}
        Certificate: {certificate}
        Duration: {runtime}
        Genres: {genre}
        IMDB Rating: {rating} based on {votes} votes
        Metacritic Score: {meta_score}
        Director: {director}
        Cast: {star1}, {star2}, {star3}, {star4}
        Box Office: {gross}
        
        Plot Summary:
        {overview}
        
        This movie belongs to the {genre} genre(s). It was directed by {director} and stars {star1}, {star2}, {star3}, and {star4}.
        It was released in {year} and has a runtime of {runtime}. It received an IMDB rating of {rating} and a Metacritic score of {meta_score}.
        The movie earned {gross} at the box office and received {votes} votes on IMDB.
        """

    def vectorize_and_upload(self, csv_path: str):
        """Vectorize IMDB data and upload to Qdrant with structured payload"""
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

        # Preprocess data
        df['runtime_minutes'] = df['Runtime'].apply(self._parse_runtime)
        df['gross_numeric'] = df['Gross'].apply(self._parse_gross)
        df['decade'] = df['Released_Year'].apply(self._get_decade)
        
        # Ensure numeric types for filtering
        df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce').fillna(0)
        df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce').fillna(0)
        df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce').fillna(0)
        df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce').fillna(0)
        
        total_records = len(df)
        
        # Process in batches
        for i in range(0, total_records, self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            batch_records = batch_df.to_dict('records')
            
            # Prepare points for upload
            points = []
            for idx, movie in enumerate(batch_records, start=i):
                # Extract data with handling for missing values
                title = str(movie.get('Series_Title', '')) if not pd.isna(movie.get('Series_Title', '')) else ''
                year = int(movie.get('Released_Year', 0)) if not pd.isna(movie.get('Released_Year', 0)) else 0
                runtime_mins = int(movie.get('runtime_minutes', 0)) if not pd.isna(movie.get('runtime_minutes', 0)) else 0
                rating = float(movie.get('IMDB_Rating', 0)) if not pd.isna(movie.get('IMDB_Rating', 0)) else 0
                meta_score = int(movie.get('Meta_score', 0)) if not pd.isna(movie.get('Meta_score', 0)) else 0
                genres = self._split_genres(movie.get('Genre', ''))
                votes = int(movie.get('No_of_Votes', 0)) if not pd.isna(movie.get('No_of_Votes', 0)) else 0
                gross = float(movie.get('gross_numeric', 0)) if not pd.isna(movie.get('gross_numeric', 0)) else 0
                decade = int(movie.get('decade', 0)) if not pd.isna(movie.get('decade', 0)) else 0
                director = str(movie.get('Director', '')) if not pd.isna(movie.get('Director', '')) else ''
                certificate = str(movie.get('Certificate', '')) if not pd.isna(movie.get('Certificate', '')) else ''
                stars = self._split_stars(movie)
                overview = str(movie.get('Overview', '')) if not pd.isna(movie.get('Overview', '')) else ''
                
                # Prepare text for embedding
                movie_text = self.prepare_movie_text(movie)
                
                try:
                    # Create embedding
                    embedding = create_embedding_with_retry(self.embeddings, movie_text)
                    
                    # Create point with structured payload
                    point = PointStruct(
                        id=idx,
                        vector=embedding,
                        payload={
                            # Full text content
                            "content": overview,
                            "full_text": movie_text,
                            
                            # Structured metadata for filtering
                            "metadata": {
                                "title": title,
                                "year": year,
                                "decade": decade,
                                "certificate": certificate,
                                "runtime_minutes": runtime_mins,
                                "genres": genres,
                                "genre_text": movie.get('Genre', ''),
                                "rating": rating,
                                "meta_score": meta_score,
                                "director": director,
                                "stars": stars,
                                "votes": votes,
                                "gross_numeric": gross,
                                "gross_text": movie.get('Gross', '')
                            }
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    print(f"Error processing record {idx} ({title}): {e}")
                    continue
            
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