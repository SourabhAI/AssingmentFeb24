import os
import streamlit as st
import time
import pandas as pd
from query_engine import IMDBQueryEngine

# Get our Streamlit page looking nice
st.set_page_config(
    page_title="IMDB Movie Chat",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    
    # Grab the settings from environment variables
    collection_name = os.environ.get('IMDB_COLLECTION_NAME', 'imdb_movies')
    llm_type = os.environ.get('IMDB_LLM_TYPE', 'gemini')
    
    try:
        # Fire up the query engine
        st.session_state.query_engine = IMDBQueryEngine(
            collection_name=collection_name,
            llm_type=llm_type
        )
        
        # Say hello when the user first arrives
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"""👋 Hey there! Welcome to the IMDB Movie Chat! By Sourabh Karandikar
            
I'm your friendly movie-buff assistant. Ask me anything about films, directors, actors - I've got the scoop on thousands of movies.

Try these questions to get started:

- When did The Matrix release?
- What are the top 5 movies of 2019 by meta score?
- Top comedy movies between 2010-2020 by IMDB rating?
- Top horror movies with a meta score above 85
- Which directors have made multiple movies that earned over 500M?
- Tell me about Steven Spielberg's best sci-fi films

What movie stuff are you curious about today?"""
        })
        
        # All good!
        st.session_state.init_success = True
    except Exception as e:
        st.error(f"Oops, hit a snag while starting up: {str(e)}")
        st.session_state.init_success = False
        st.session_state.init_error = str(e)

# Create a nice sidebar with info about the app
with st.sidebar:
    st.title("🎬 IMDB Movie Chat")
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    Hey! This app lets you chat about movies in the IMDB database.
    
    **Cool features:**
    - Ask about specific movies, directors, or actors
    - Filter by year, rating, genre - whatever you want
    - Get ranked lists of top movies
    - Deep dive into directors' filmographies
    
    Under the hood, I'm using vector search to find relevant movies and AI to craft helpful answers.
    """)
    
    # Only show sample queries if everything's working
    if st.session_state.get("init_success", False):
        st.markdown("---")
        st.subheader("Try These Questions")
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
        
        # Add clickable examples
        for query in sample_queries:
            if st.button(f"{query}", key=f"sample_{query[:20]}"):
                # Add user query to chat
                if "messages" in st.session_state:
                    st.session_state.messages.append({"role": "user", "content": query})
                
                # Safely refresh the page
                st.experimental_rerun()

# Main chat area
st.title("🎬 IMDB Movie Chat")

# Show an error if something went wrong during startup
if not st.session_state.get("init_success", False):
    st.error(f"Something's not right: {st.session_state.get('init_error', 'Unknown error')}")
    st.warning("Make sure Qdrant is running and your API keys are set correctly.")
    st.stop()  # Stop here if we can't proceed

# Show the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new messages from the user
if prompt := st.chat_input("What movie question can I answer for you?"):
    # Add the user's message to the chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Now generate and show the response
    with st.chat_message("assistant"):
        with st.spinner("Searching for movie info..."):
            message_placeholder = st.empty()
            full_response = ""
            
            # Process the user's question
            try:
                result = st.session_state.query_engine.query(prompt)
                if result and "answer" in result:
                    answer = result["answer"]
                    
                    # Add a note about filtered results
                    if result.get("filtered_search"):
                        answer += f"\n\n_Found this from {result.get('num_results', 'several')} filtered results_"
                    
                    # Add a typing effect to make it feel more natural
                    words = answer.split()
                    for i in range(0, len(words), 2):  # Process 2 words at a time to be more efficient
                        chunk = " ".join(words[i:i+2])
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.005)  # Quick typing but still visible
                    
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.markdown("I'm not sure how to answer that one. Could you try rephrasing or asking about a specific movie?")
            except Exception as e:
                error_message = f"Oops! Something went wrong: {str(e)}\n\nTry asking a different way or check that the movie database is working correctly."
                message_placeholder.markdown(error_message)
                full_response = error_message
    
    # Save the response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response}) 