import os
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue, MatchAny
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class LLMFactory:
    """Factory class to create different LLM instances"""
    
    @staticmethod
    def create_llm(llm_type: str = "gemini", **kwargs):
        if llm_type == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7,
                **kwargs
            )
        elif llm_type == "openai":
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
                **kwargs
            )
        elif llm_type == "llama":
            return LlamaCpp(
                model_path=kwargs.get("model_path", "models/llama-2-7b-chat.gguf"),
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

class QueryParser:
    """Parser for extracting filtering criteria from natural language queries"""

    @staticmethod
    def extract_year_range(query: str) -> Optional[Tuple[int, int]]:
        """Extract year range from query text (e.g., 'between 2010 and 2020', 'from 2010 to 2020', '2010-2020')"""
        # Pattern matching for various year range formats
        patterns = [
            r'(?:between|from)?\s*(\d{4})\s*(?:[-–—]|to|and)\s*(\d{4})',  # 2010-2020, from 2010 to 2020, between 2010 and 2020
            r'(?:in|from|since)\s*(\d{4})\s*(?:to present|until now|to today)'  # from 2010 to present
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2:
                        year1, year2 = int(match.group(1)), int(match.group(2))
                        return min(year1, year2), max(year1, year2)
                    else:
                        # For "from X to present", use current year as end
                        from datetime import datetime
                        current_year = datetime.now().year
                        return int(match.group(1)), current_year
                except (ValueError, IndexError):
                    continue
        
        # Look for a single year - expanded to catch more patterns including "of 2019"
        year_match = re.search(r'(?:in|from|of|year|during|released?(?:\sin)?)\s*(\d{4})', query, re.IGNORECASE)
        if year_match:
            try:
                year = int(year_match.group(1))
                return year, year
            except (ValueError, IndexError):
                pass
        
        # Direct check for 4-digit years in the query as a fallback
        year_digits = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        if year_digits:
            try:
                year = int(year_digits[0])
                return year, year
            except (ValueError, IndexError):
                pass
        
        return None

    @staticmethod
    def extract_multiplicity(query: str) -> Optional[int]:
        """Extract multiplicity requirements like 'at least twice', 'more than 3 times', etc."""
        patterns = [
            r'at least (\d+) times?',
            r'at least (\w+) times?',
            r'more than (\d+) times?',
            r'more than (\w+) times?',
            r'(\d+)\+ times?',
            # Add patterns that handle specific contexts like "at least twice" at the end of a phrase
            r'(?:with|having|earning).*?at least (\w+)',
            r'(?:with|having|earning).*?at least (\d+)',
            r'greater than .*? at least (\w+)',
            r'greater than .*? at least (\d+)',
            r'more than .*? at least (\w+)',
            r'more than .*? at least (\d+)',
            # Specific patterns for phrases like "earnings of greater than 500M at least twice"
            r'(?:gross|earnings|box office|revenue).*?greater than.*?at least (\w+)',
            r'(?:gross|earnings|box office|revenue).*?greater than.*?at least (\d+)'
        ]
        
        word_to_number = {
            'once': 1, 'twice': 2, 'thrice': 3,
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).lower()
                    if value.isdigit():
                        return int(value)
                    elif value in word_to_number:
                        return word_to_number[value]
                except (ValueError, IndexError):
                    continue
        
        # Special case for "twice" without "at least"
        if re.search(r'\btwice\b', query, re.IGNORECASE):
            return 2
            
        return None

    @staticmethod
    def is_director_analysis_query(query: str) -> bool:
        """Detect queries that require analysis of directors and their movies"""
        patterns = [
            r'(?:top|best) directors',
            r'directors with',
            r'directors who',
            r'directors and their',
            # Add specific pattern for our example query
            r'directors.*?(?:highest|best|top).*?(?:gross|earning)',
            r'(?:top|best).*?directors.*?(?:movies|films)',
            r'(?:top|best).*?directors.*?(?:gross|earning|revenue|box office)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False

    @staticmethod
    def extract_rating_range(query: str) -> Optional[Tuple[float, float]]:
        """Extract IMDB rating range from query text (e.g., 'rating above 8.5', 'rating between 7 and 9')"""
        # Pattern for rating above/below threshold
        above_pattern = r'rating\s+(?:above|greater than|higher than|more than|>\s*|≥\s*|>=\s*)(\d+\.?\d*)'
        below_pattern = r'rating\s+(?:below|less than|lower than|<\s*|≤\s*|<=\s*)(\d+\.?\d*)'
        between_pattern = r'rating\s+(?:between|from)\s+(\d+\.?\d*)\s+(?:and|to)\s+(\d+\.?\d*)'
        
        # Check for "between" pattern first
        between_match = re.search(between_pattern, query, re.IGNORECASE)
        if between_match:
            try:
                rating1 = float(between_match.group(1))
                rating2 = float(between_match.group(2))
                return min(rating1, rating2), max(rating1, rating2)
            except (ValueError, IndexError):
                pass
        
        # Check for above/below patterns
        above_match = re.search(above_pattern, query, re.IGNORECASE)
        below_match = re.search(below_pattern, query, re.IGNORECASE)
        
        min_rating = 0.0
        max_rating = 10.0
        
        if above_match:
            try:
                min_rating = float(above_match.group(1))
            except (ValueError, IndexError):
                pass
        
        if below_match:
            try:
                max_rating = float(below_match.group(1))
            except (ValueError, IndexError):
                pass
        
        if above_match or below_match:
            return min_rating, max_rating
        
        return None

    @staticmethod
    def extract_meta_score_range(query: str) -> Optional[Tuple[int, int]]:
        """Extract meta score range from query text (e.g., 'meta score above 85', 'meta score between 70 and 90')"""
        # Pattern for meta score above/below threshold
        above_pattern = r'meta\s*(?:score|critic)?\s+(?:above|greater than|higher than|more than|>\s*|≥\s*|>=\s*)(\d+)'
        below_pattern = r'meta\s*(?:score|critic)?\s+(?:below|less than|lower than|<\s*|≤\s*|<=\s*)(\d+)'
        between_pattern = r'meta\s*(?:score|critic)?\s+(?:between|from)\s+(\d+)\s+(?:and|to)\s+(\d+)'
        
        # Check for "between" pattern first
        between_match = re.search(between_pattern, query, re.IGNORECASE)
        if between_match:
            try:
                score1 = int(between_match.group(1))
                score2 = int(between_match.group(2))
                return min(score1, score2), max(score1, score2)
            except (ValueError, IndexError):
                pass
        
        # Check for above/below patterns
        above_match = re.search(above_pattern, query, re.IGNORECASE)
        below_match = re.search(below_pattern, query, re.IGNORECASE)
        
        min_score = 0
        max_score = 100
        
        if above_match:
            try:
                min_score = int(above_match.group(1))
            except (ValueError, IndexError):
                pass
        
        if below_match:
            try:
                max_score = int(below_match.group(1))
            except (ValueError, IndexError):
                pass
        
        if above_match or below_match:
            return min_score, max_score
        
        return None

    @staticmethod
    def extract_genres(query: str) -> List[str]:
        """Extract genre information from query text"""
        # Common movie genres
        known_genres = [
            "Action", "Adventure", "Animation", "Biography", "Comedy", 
            "Crime", "Documentary", "Drama", "Family", "Fantasy", 
            "Film-Noir", "History", "Horror", "Music", "Musical", 
            "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", 
            "War", "Western"
        ]
        
        # Find all genres mentioned in the query
        found_genres = []
        for genre in known_genres:
            # Look for the genre as a whole word
            if re.search(r'\b' + re.escape(genre) + r'\b', query, re.IGNORECASE):
                found_genres.append(genre)
                
        return found_genres

    @staticmethod
    def extract_directors(query: str) -> List[str]:
        """Extract director names from query text"""
        # Look for patterns like "directed by X", "director X", "X's movies"
        director_patterns = [
            r'directed by\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            r'director\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(?:directed|movies|films)',
            r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)'s\s+(?:movies|films)"
        ]
        
        directors = []
        for pattern in director_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                directors.append(match.group(1).strip())
                
        return directors

    @staticmethod
    def extract_actors(query: str) -> List[str]:
        """Extract actor names from query text"""
        # Look for patterns like "starring X", "with X", "featuring X", "X's movies"
        actor_patterns = [
            r'starring\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            r'with\s+(?:actor|actress)?\s*([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            r'featuring\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            r'(?:actor|actress)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)'
        ]
        
        actors = []
        for pattern in actor_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                actors.append(match.group(1).strip())
                
        return actors

    @staticmethod
    def extract_votes_threshold(query: str) -> Optional[int]:
        """Extract vote count threshold from query text (e.g., 'over 1M votes', 'more than 500000 votes')"""
        # Pattern for votes above threshold
        vote_patterns = [
            r'(?:with|having|over|more than)\s+(\d+)([Kk]|[Mm])?\s+votes',
            r'(?:votes|vote count)\s+(?:above|greater than|more than|>\s*|≥\s*|>=\s*)(\d+)([Kk]|[Mm])?'
        ]
        
        for pattern in vote_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) >= 2:
                        count = int(match.group(1))
                        multiplier = match.group(2)
                        
                        if multiplier:
                            if multiplier.lower() == 'k':
                                count *= 1000
                            elif multiplier.lower() == 'm':
                                count *= 1000000
                                
                        return count
                except (ValueError, IndexError):
                    continue
        
        return None

    @staticmethod
    def extract_gross_threshold(query: str) -> Optional[float]:
        """Extract gross earnings threshold from query text (e.g., 'over $500M', 'earnings above 100 million')"""
        # Pattern for gross above threshold
        gross_patterns = [
            r'(?:gross|earnings|box office|revenue)\s+(?:above|greater than|more than|>\s*|≥\s*|>=\s*)\s*\$?\s*(\d+(?:\.\d+)?)\s*([Kk]|[Mm]|[Bb]|million|billion)?',
            r'(?:gross|earnings|box office|revenue)\s+(?:of|over|exceeding)\s+\$?\s*(\d+(?:\.\d+)?)\s*([Kk]|[Mm]|[Bb]|million|billion)?'
        ]
        
        for pattern in gross_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1))
                    multiplier = match.group(2) if len(match.groups()) >= 2 else None
                    
                    if multiplier:
                        if multiplier.lower() in ['m', 'million']:
                            amount *= 1000000
                        elif multiplier.lower() in ['b', 'billion']:
                            amount *= 1000000000
                        elif multiplier.lower() == 'k':
                            amount *= 1000
                            
                    return amount
                except (ValueError, IndexError):
                    continue
        
        return None

    @staticmethod
    def extract_top_n(query: str) -> Optional[int]:
        """Extract 'top N' from query (e.g., 'top 5 movies', 'top 10 highest rated')"""
        pattern = r'top\s+(\d+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        
        return None

    @staticmethod
    def parse_query(query: str) -> Dict[str, Any]:
        """Parse a natural language query to extract filtering criteria"""
        filters = {}
        
        # Extract year range
        year_range = QueryParser.extract_year_range(query)
        if year_range:
            filters['year_range'] = year_range
        
        # Extract rating range
        rating_range = QueryParser.extract_rating_range(query)
        if rating_range:
            filters['rating_range'] = rating_range
        
        # Extract meta score range
        meta_score_range = QueryParser.extract_meta_score_range(query)
        if meta_score_range:
            filters['meta_score_range'] = meta_score_range
        
        # Extract genres
        genres = QueryParser.extract_genres(query)
        if genres:
            filters['genres'] = genres
        
        # Extract directors
        directors = QueryParser.extract_directors(query)
        if directors:
            filters['directors'] = directors
        
        # Extract actors
        actors = QueryParser.extract_actors(query)
        if actors:
            filters['actors'] = actors
        
        # Extract votes threshold
        votes_threshold = QueryParser.extract_votes_threshold(query)
        if votes_threshold:
            filters['votes_min'] = votes_threshold
        
        # Extract gross threshold
        gross_threshold = QueryParser.extract_gross_threshold(query)
        if gross_threshold:
            filters['gross_min'] = gross_threshold
        
        # Extract multiplicity requirements
        multiplicity = QueryParser.extract_multiplicity(query)
        if multiplicity:
            filters['multiplicity'] = multiplicity
            
        # Detect special query types
        if QueryParser.is_director_analysis_query(query):
            filters['query_type'] = 'director_analysis'
        
        # Extract top N
        top_n = QueryParser.extract_top_n(query)
        if top_n:
            filters['top_n'] = top_n
        
        return filters

    @staticmethod
    def _extract_movie_title(query: str) -> Optional[str]:
        """Extract potential movie title from a query."""
        # Pattern for movie titles in various question formats
        patterns = [
            # "When did {Movie} release?"
            r"when did (?:the\s+)?([A-Z][a-zA-Z0-9\s']+?)(?:\s+release|\s+come out|\s+debut)",
            # "Tell me about {Movie}"
            r"tell me about (?:the\s+)?([A-Z][a-zA-Z0-9\s']+)",
            # "What is {Movie} about?"
            r"what is (?:the\s+)?([A-Z][a-zA-Z0-9\s']+) about",
            # "Who directed {Movie}?"
            r"who directed (?:the\s+)?([A-Z][a-zA-Z0-9\s']+)",
            # "Who starred in {Movie}?"
            r"who (?:starred|acted) in (?:the\s+)?([A-Z][a-zA-Z0-9\s']+)",
            # "{Movie} cast"
            r"(?:the\s+)?([A-Z][a-zA-Z0-9\s']+) cast",
            # "What year was {Movie} released?"
            r"what year was (?:the\s+)?([A-Z][a-zA-Z0-9\s']+) released",
            # Just look for capitalized phrases that might be titles
            r"\b([A-Z][a-zA-Z0-9\s']+)\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Check if it looks like a title (avoid extracting general phrases)
                if len(title.split()) <= 6 and not title.lower() in ["the movie", "the film", "it", "that"]:
                    return title
        
        return None

class IMDBQueryEngine:
    """Advanced query engine for IMDB movie data with filtering capabilities"""
    
    def __init__(
        self,
        collection_name: str = "imdb_movies",
        llm_type: str = "gemini",
        **llm_kwargs
    ):
        # Initialize vector store
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings,
            content_payload_key="full_text",  # Use full_text field as document content
            metadata_payload_key="metadata"   # Use metadata field for document metadata
        )
        
        # Initialize LLM and chain
        self.llm = LLMFactory.create_llm(llm_type, **llm_kwargs)
        self.qa_chain = self._create_qa_chain()
        
        # Initialize query parser
        self.parser = QueryParser()
        
    def _create_qa_chain(self) -> RetrievalQA:
        prompt_template = """You are a helpful movie expert assistant. Use the following context to answer the question.
        If you don't have enough information, just say so.
        
        Context: {context}
        
        Question: {query}
        
        Provide a detailed and accurate answer based on the context provided. If relevant, include:
        - Movie titles
        - Release years
        - Directors
        - Ratings
        - Box office numbers
        - Other relevant details from the context
        
        If you need to sort or rank items, make sure to explain the criteria used.
        If the question asks for a summary, provide a concise but informative summary.
        If the question asks for comparisons, highlight key similarities and differences.
        
        If you don't have enough information, please say so rather than making assumptions.
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )
        
        try:
            # Create chain with error handling
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}  # Increased from 5 to 10
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False
            )
        except Exception as e:
            print(f"Warning: Error creating QA chain: {e}. Will use fallback methods instead.")
            # Return a minimal chain that will be bypassed anyway
            return None

    def _build_filter(self, filter_criteria: Dict[str, Any]) -> Optional[Filter]:
        """Build a Qdrant filter from parsed filter criteria"""
        if not filter_criteria:
            return None
            
        conditions = []
        
        # Year range filter
        if 'year_range' in filter_criteria:
            min_year, max_year = filter_criteria['year_range']
            
            # If exact year (min=max), make it very clear
            if min_year == max_year:
                print(f"Filtering for exact year: {min_year}")
                conditions.append(
                    FieldCondition(
                        key="metadata.year",
                        range=Range(
                            gte=min_year,
                            lte=min_year
                        )
                    )
                )
            else:
                print(f"Filtering for year range: {min_year}-{max_year}")
                conditions.append(
                    FieldCondition(
                        key="metadata.year",
                        range=Range(
                            gte=min_year,
                            lte=max_year
                        )
                    )
                )
        
        # Rating range filter
        if 'rating_range' in filter_criteria:
            min_rating, max_rating = filter_criteria['rating_range']
            conditions.append(
                FieldCondition(
                    key="metadata.rating",
                    range=Range(
                        gte=min_rating,
                        lte=max_rating
                    )
                )
            )
        
        # Meta score range filter
        if 'meta_score_range' in filter_criteria:
            min_score, max_score = filter_criteria['meta_score_range']
            conditions.append(
                FieldCondition(
                    key="metadata.meta_score",
                    range=Range(
                        gte=min_score,
                        lte=max_score
                    )
                )
            )
        
        # Votes threshold filter
        if 'votes_min' in filter_criteria:
            conditions.append(
                FieldCondition(
                    key="metadata.votes",
                    range=Range(
                        gte=filter_criteria['votes_min']
                    )
                )
            )
        
        # Gross threshold filter
        if 'gross_min' in filter_criteria:
            conditions.append(
                FieldCondition(
                    key="metadata.gross_numeric",
                    range=Range(
                        gte=filter_criteria['gross_min']
                    )
                )
            )
        
        # Genre filter
        if 'genres' in filter_criteria and filter_criteria['genres']:
            conditions.append(
                FieldCondition(
                    key="metadata.genres",
                    match=MatchAny(
                        any=filter_criteria['genres']
                    )
                )
            )
        
        # Director filter
        if 'directors' in filter_criteria and filter_criteria['directors']:
            # For demo purposes, just use the first director mentioned
            if len(filter_criteria['directors']) == 1:
                conditions.append(
                    FieldCondition(
                        key="metadata.director",
                        match=MatchValue(
                            value=filter_criteria['directors'][0]
                        )
                    )
                )
            else:
                # If multiple directors, search full text instead (less precise)
                # We'll rely on semantic search for this
                pass
        
        # Actor filter (this is approximate as actors are in the stars array)
        if 'actors' in filter_criteria and filter_criteria['actors']:
            for actor in filter_criteria['actors']:
                conditions.append(
                    FieldCondition(
                        key="metadata.stars",
                        match=MatchValue(
                            value=actor
                        )
                    )
                )
        
        # If no conditions, return None (no filtering)
        if not conditions:
            return None
        
        # Return filter with all conditions (AND logic)
        return Filter(
            must=conditions
        )

    def search_with_filter(self, query_text: str, filter_criteria: Dict[str, Any], limit: int = 10) -> List[Dict]:
        """Search Qdrant with filtering and return raw results"""
        # Create query embedding
        query_vector = self.embeddings.embed_query(query_text)
        
        # Build filter
        qdrant_filter = self._build_filter(filter_criteria)
        print(f"Using Qdrant filter: {qdrant_filter}")
        
        # Determine sorting parameters for "top N" queries
        score_key = None
        if 'rating_range' in filter_criteria:
            score_key = "metadata.rating"
        elif 'meta_score_range' in filter_criteria:
            score_key = "metadata.meta_score"
        elif 'gross_min' in filter_criteria:
            score_key = "metadata.gross_numeric"
        elif 'votes_min' in filter_criteria:
            score_key = "metadata.votes"
        
        # If query contains "by meta score", prioritize meta score sorting
        if "by meta score" in query_text.lower():
            score_key = "metadata.meta_score"
            print("Detected 'by meta score' in query, sorting by meta score")
        elif "by rating" in query_text.lower() or "by imdb rating" in query_text.lower():
            score_key = "metadata.rating"
            print("Detected rating sort request in query, sorting by IMDB rating")
        
        # Get top_n if specified
        top_n = filter_criteria.get('top_n', limit)
        
        # Perform search
        search_result = self.qdrant_client.search(
            collection_name=self.vector_store.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=max(top_n * 2, 20),  # Get more results than needed to account for scoring
            with_payload=True,
            with_vectors=False
        )
        
        # Process results
        results = []
        for scored_point in search_result:
            payload = scored_point.payload
            results.append({
                "score": scored_point.score,
                "payload": payload
            })
            
        # If we have a score_key, re-rank results
        if score_key and len(results) > 0:
            if score_key == "metadata.rating" or score_key == "metadata.meta_score" or score_key == "metadata.votes" or score_key == "metadata.gross_numeric":
                # Sort by specified field (descending)
                print(f"Re-ranking results by {score_key}")
                try:
                    # Get nested value safely
                    results.sort(
                        key=lambda x: float(self._get_nested_dict_value(x["payload"], score_key.split('.')) or 0),
                        reverse=True
                    )
                    # Print top 3 items with their keys
                    for i, item in enumerate(results[:3]):
                        value = self._get_nested_dict_value(item["payload"], score_key.split('.'))
                        title = item["payload"].get("metadata", {}).get("title", "Unknown")
                        print(f"  Rank {i+1}: {title} with {score_key.split('.')[-1]}={value}")
                except (KeyError, TypeError, ValueError) as e:
                    # Fallback to similarity score if field not found
                    print(f"Error sorting by {score_key}: {e}")
                    pass
        
        # Limit to requested number
        return results[:top_n]

    @staticmethod
    def _get_nested_dict_value(d: Dict, keys: List[str]) -> Any:
        """Helper to safely get nested dictionary values"""
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return None
        return d

    def prepare_context_from_results(self, results: List[Dict]) -> str:
        """Format search results into context for the LLM"""
        if not results:
            return "No relevant movies found."
            
        context_parts = ["Here is information about relevant movies:"]
        
        for i, result in enumerate(results, 1):
            payload = result["payload"]
            metadata = payload.get("metadata", {})
            
            # Extract year and make it prominent
            year = metadata.get('year', 'Unknown Year')
            
            # Build a structured representation of the movie
            movie_info = [
                f"Movie {i}: {metadata.get('title', 'Unknown Title')} ({year})",
                f"Director: {metadata.get('director', 'Unknown')}",
                f"Genre(s): {metadata.get('genre_text', 'Unknown')}",
                f"IMDB Rating: {metadata.get('rating', 'Unknown')} (from {metadata.get('votes', 'Unknown')} votes)",
                f"Meta Score: {metadata.get('meta_score', 'Unknown')}",
                f"Box Office: {metadata.get('gross_text', 'Unknown')}",
                f"Cast: {', '.join(metadata.get('stars', []))}"
            ]
            
            # Add plot if available
            if "content" in payload:
                movie_info.append(f"Plot: {payload['content']}")
                
            # Add to context
            context_parts.append("\n".join(movie_info))
            
        # Join all movie information with blank lines in between
        context = "\n\n".join(context_parts)
        
        # Debugging: Log some information about the years in the results
        years = [result["payload"].get("metadata", {}).get("year", "Unknown") for result in results]
        print(f"Years of movies in results: {years}")
        
        return context

    def query(self, query: str) -> Dict:
        """Process a single question with advanced filtering and RAG"""
        try:
            print("Parsing query filters...")
            # Parse query to extract filter criteria
            filter_criteria = self.parser.parse_query(query)
            print(f"Extracted filters: {json.dumps(filter_criteria, indent=2)}")
            
            # Handle special query types that require multi-step processing
            if filter_criteria.get('query_type') == 'director_analysis':
                print("Handling special director analysis query...")
                return self._handle_director_analysis_query(query, filter_criteria)
            
            if filter_criteria:
                print("Applying filters to search...")
                # Get results with filtering
                search_results = self.search_with_filter(query, filter_criteria)
                
                # Prepare context from filtered results
                context = self.prepare_context_from_results(search_results)
                
                # If we found results, use them directly with the LLM
                if search_results:
                    print(f"Found {len(search_results)} results after filtering")
                    # Call LLM with prepared context
                    prompt_text = f"""Answer this question based on the provided context.
                        
                        Context:
                        {context}
                        
                        Question:
                        {query}
                        
                        Provide a detailed and accurate answer. Include relevant titles, ratings, and other movie information as appropriate.
                        """
                    
                    if isinstance(self.llm, (ChatGoogleGenerativeAI, ChatOpenAI)):
                        response = self.llm.invoke([{"role": "user", "content": prompt_text}])
                    else:
                        # For other LLM types like LlamaCpp
                        response = self.llm.invoke(prompt_text)
                    
                    return {
                        "answer": response.content,
                        "filtered_search": True,
                        "num_results": len(search_results)
                    }
            
            print("Using direct search approach instead of RetrievalQA chain...")
            try:
                # Direct search without using RetrievalQA chain to avoid document validation issues
                docs = self.vector_store.similarity_search(query, k=10)
                
                if not docs:
                    return {
                        "answer": "I couldn't find any relevant information to answer your question.",
                        "filtered_search": False
                    }
                
                # Create context from valid documents
                context_parts = []
                for doc in docs:
                    if hasattr(doc, 'page_content') and doc.page_content:
                        context_parts.append(doc.page_content)
                
                context = "\n\n".join(context_parts)
                
                if not context.strip():
                    return {
                        "answer": "I found information but couldn't process it properly. Please try a more specific question.",
                        "filtered_search": False
                    }
                
                # Use LLM directly with the prompt
                prompt_text = f"""You are a helpful movie expert assistant. Use the following context to answer the question.
                If you don't have enough information, just say so.
                
                Context: {context}
                
                Question: {query}
                
                Provide a detailed and accurate answer based on the context provided.
                """
                
                if isinstance(self.llm, (ChatGoogleGenerativeAI, ChatOpenAI)):
                    response = self.llm.invoke([{"role": "user", "content": prompt_text}])
                else:
                    # For other LLM types like LlamaCpp
                    response = self.llm.invoke(prompt_text)
                
                return {
                    "answer": response.content,
                    "filtered_search": False
                }
                
            except Exception as chain_error:
                print(f"Error in direct search: {chain_error}")
                # Final fallback - search for movie by title using regex
                try:
                    # Try to extract possible movie title using our helper
                    possible_title = QueryParser._extract_movie_title(query)
                    
                    if possible_title:
                        print(f"Extracted possible title: '{possible_title}'")
                        
                        # Search using extracted title
                        results = self.qdrant_client.search(
                            collection_name=self.vector_store.collection_name,
                            query_vector=self.embeddings.embed_query(possible_title),
                            limit=5,
                            with_payload=True
                        )
                        
                        if results:
                            context = self.prepare_context_from_results([
                                {"payload": result.payload} for result in results
                            ])
                            
                            prompt_text = f"""Answer this question about a movie based on the provided context.
                                
                                Context: {context}
                                
                                Question: {query}
                                
                                If the context doesn't contain the movie mentioned in the question, please say so clearly.
                            """
                            
                            if isinstance(self.llm, (ChatGoogleGenerativeAI, ChatOpenAI)):
                                response = self.llm.invoke([{"role": "user", "content": prompt_text}])
                            else:
                                response = self.llm.invoke(prompt_text)
                            
                            return {
                                "answer": response.content,
                                "filtered_search": False
                            }
                    
                    # If no title found or no results, perform a general search
                    print("No specific movie title found, using general search")
                    results = self.qdrant_client.search(
                        collection_name=self.vector_store.collection_name,
                        query_vector=self.embeddings.embed_query(query),
                        limit=5,
                        with_payload=True
                    )
                    
                    if results:
                        context = self.prepare_context_from_results([
                            {"payload": result.payload} for result in results
                        ])
                        
                        prompt_text = f"""Answer this movie-related question based on the provided context.
                            
                            Context: {context}
                            
                            Question: {query}
                            
                            If you don't have enough information to answer the question precisely, please say so.
                        """
                        
                        if isinstance(self.llm, (ChatGoogleGenerativeAI, ChatOpenAI)):
                            response = self.llm.invoke([{"role": "user", "content": prompt_text}])
                        else:
                            response = self.llm.invoke(prompt_text)
                        
                        return {
                            "answer": response.content,
                            "filtered_search": False
                        }
                    
                    return {
                        "answer": "I couldn't find an answer to that question. Please try rephrasing or asking about a specific movie.",
                        "filtered_search": False
                    }
                
                except Exception as final_error:
                    print(f"Final fallback error: {final_error}")
                    return {
                        "answer": "I encountered an error searching for information. Please try asking about a specific movie title more directly."
                    }
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return {
                "answer": "Sorry, I encountered an error. Please try rephrasing your question with more specific movie details."
            }

    def _handle_director_analysis_query(self, query: str, filter_criteria: Dict[str, Any]) -> Dict:
        """Handle queries that analyze directors and their movies"""
        try:
            print("Fetching data for director analysis...")
            # Get all movies first - we need a large dataset for aggregation
            all_points = self.qdrant_client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=1000,  # Get up to 1000 movies
                with_payload=True,
                with_vectors=False
            )[0]  # scroll returns (points, next_page_offset)
            
            print(f"Retrieved {len(all_points)} movies for analysis")
            
            # Group by director
            directors = {}
            for point in all_points:
                payload = point.payload
                metadata = payload.get("metadata", {})
                
                director = metadata.get("director")
                if not director:
                    continue
                    
                # Handle gross values that might be stored in different formats
                gross = 0
                gross_numeric = metadata.get("gross_numeric")
                
                # Try to get numeric gross value
                if gross_numeric is not None:
                    # It might be stored as a number
                    if isinstance(gross_numeric, (int, float)):
                        gross = float(gross_numeric)
                    # Or it might be stored as a string that can be converted
                    elif isinstance(gross_numeric, str) and gross_numeric.strip():
                        try:
                            gross = float(gross_numeric)
                        except ValueError:
                            pass
                
                # If we still don't have a gross value, try to parse it from gross_text
                if gross == 0:
                    gross_text = metadata.get("gross_text", "")
                    if gross_text and isinstance(gross_text, str):
                        # Extract numbers from strings like "$100.5M" or "$1.2B"
                        match = re.search(r'\$?([\d,.]+)([KMB])?', gross_text)
                        if match:
                            try:
                                amount = float(match.group(1).replace(',', ''))
                                unit = match.group(2) if len(match.groups()) > 1 else None
                                
                                if unit:
                                    if unit.upper() == 'K':
                                        amount *= 1000
                                    elif unit.upper() == 'M':
                                        amount *= 1000000
                                    elif unit.upper() == 'B':
                                        amount *= 1000000000
                                        
                                gross = amount
                            except ValueError:
                                pass
                
                # Skip if no valid gross found
                if gross <= 0:
                    continue
                    
                if director not in directors:
                    directors[director] = []
                    
                directors[director].append({
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", "Unknown"),
                    "gross": gross,
                    "rating": metadata.get("rating", 0),
                    "meta_score": metadata.get("meta_score", 0),
                    "payload": payload
                })
            
            # Get gross threshold from filter criteria
            gross_threshold = filter_criteria.get('gross_min', 500000000)  # Default to 500M if not specified
            print(f"Using gross threshold: ${gross_threshold:,.2f}")
            
            # Get multiplicity requirement (how many movies should meet the threshold)
            min_count = filter_criteria.get('multiplicity', 2)  # Default to 2 if not specified ("at least twice")
            print(f"Looking for directors with at least {min_count} movies above threshold")
            
            # Filter directors based on criteria
            qualifying_directors = {}
            for director, movies in directors.items():
                # Count movies that meet the gross threshold
                high_grossing_movies = [m for m in movies if m["gross"] >= gross_threshold]
                
                if len(high_grossing_movies) >= min_count:
                    # Sort movies by gross (descending)
                    sorted_movies = sorted(movies, key=lambda m: m["gross"], reverse=True)
                    qualifying_directors[director] = sorted_movies
            
            print(f"Found {len(qualifying_directors)} qualifying directors")
            
            # Get top directors by number of qualifying movies or total gross
            top_directors = sorted(
                qualifying_directors.items(),
                key=lambda x: sum(m["gross"] for m in x[1]),  # Sort by total gross
                reverse=True
            )[:10]  # Limit to top 10 directors
            
            # Prepare response
            context_parts = ["Here is information about top directors and their highest grossing movies:"]
            
            for director, movies in top_directors:
                # Get highest grossing movie
                highest_grossing = movies[0]
                
                # Get number of movies above threshold
                above_threshold_count = len([m for m in movies if m["gross"] >= gross_threshold])
                
                director_info = [
                    f"Director: {director}",
                    f"Highest Grossing Movie: {highest_grossing['title']} ({highest_grossing['year']}) - ${highest_grossing['gross']:,.2f}",
                    f"Number of Movies with Gross > ${gross_threshold:,.2f}: {above_threshold_count}",
                    f"Top Movies:"
                ]
                
                # Add top 3 movies by gross
                for i, movie in enumerate(movies[:3], 1):
                    director_info.append(f"  {i}. {movie['title']} ({movie['year']}) - ${movie['gross']:,.2f}")
                    
                context_parts.append("\n".join(director_info))
            
            context = "\n\n".join(context_parts)
            
            # Call LLM with prepared context
            prompt_text = f"""Answer this question about movie directors based on the provided information.
                
                Information:
                {context}
                
                Question:
                {query}
                
                Provide a detailed and accurate answer. Mention specific directors, their highest grossing movies, and other relevant details.
                """
            
            if isinstance(self.llm, (ChatGoogleGenerativeAI, ChatOpenAI)):
                response = self.llm.invoke([{"role": "user", "content": prompt_text}])
            else:
                # For other LLM types like LlamaCpp
                response = self.llm.invoke(prompt_text)
            
            return {
                "answer": response.content,
                "filtered_search": True,
                "num_results": len(top_directors)
            }
            
        except Exception as e:
            print(f"Error in director analysis: {e}")
            return self._fallback_to_general_search(query)
    
    def _fallback_to_general_search(self, query: str) -> Dict:
        """Fallback to general search when special handling fails"""
        print("Falling back to general search...")
        try:
            # Direct search without using RetrievalQA chain
            docs = self.vector_store.similarity_search(query, k=10)
            
            if not docs:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "filtered_search": False
                }
            
            # Create context from valid documents
            context_parts = []
            for doc in docs:
                if hasattr(doc, 'page_content') and doc.page_content:
                    context_parts.append(doc.page_content)
            
            context = "\n\n".join(context_parts)
            
            if not context.strip():
                return {
                    "answer": "I found information but couldn't process it properly. Please try a more specific question.",
                    "filtered_search": False
                }
            
            # Use LLM directly with the prompt
            prompt_text = f"""You are a helpful movie expert assistant. Use the following context to answer the question.
            If you don't have enough information, just say so.
            
            Context: {context}
            
            Question: {query}
            
            Provide a detailed and accurate answer based on the context provided.
            """
            
            if isinstance(self.llm, (ChatGoogleGenerativeAI, ChatOpenAI)):
                response = self.llm.invoke([{"role": "user", "content": prompt_text}])
            else:
                # For other LLM types like LlamaCpp
                response = self.llm.invoke(prompt_text)
            
            return {
                "answer": response.content,
                "filtered_search": False
            }
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return {
                "answer": "I encountered an error searching for information. Please try a simpler question."
            } 