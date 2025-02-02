import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime
import time
import streamlit.components.v1 as components
from add_ga import inject_ga
from db import log_query
from bson import ObjectId


# Near the top of the file, after imports
def is_dev_mode():
    """Check if the app is running in development mode"""
    return os.getenv("ENV") == "dev"


# Update the page config
st.set_page_config(
    page_title="Greenburgh Genie",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded" if is_dev_mode() else "collapsed",
)

# Add custom CSS to hide elements in production
if not is_dev_mode():
    st.markdown(
        """
        <style>
            [data-testid="collapsedControl"] {
                display: none
            }
            #MainMenu {
                visibility: hidden;
            }
            .stDeployButton {
                display: none !important;
            }
            footer {
                visibility: hidden;
            }
            /* Show sidebar in production for disclaimer */
            section[data-testid="stSidebar"] {
                width: 300px !important;
                background-color: #f8f9f9;
            }
            /* Adjust the main content area */
            .main > div {
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 64rem;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

# Load environment variables
load_dotenv()

# Initialize Pinecone with new syntax
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Pinecone index with new syntax
index = pc.Index(os.getenv("PINECONE_INDEX"))


# Near the top of the file, after initializing Pinecone
def optimize_index():
    """
    Configure index for optimal performance
    """
    try:
        # Get current index description
        index_description = index.describe_index_stats()

        # Set pod type for better performance if needed
        # Note: This might require index recreation
        index_config = {
            "pod_type": "p1.x1",  # Use high-performance pod
            "pods": 1,
            "replicas": 1,
            "metadata_config": {
                "indexed": ["source", "timestamp"]  # Index metadata fields
            },
        }

        if is_dev_mode():
            st.sidebar.write("### Index Configuration")
            st.sidebar.write(f"Pod Type: {index_config['pod_type']}")
            st.sidebar.write(f"Vectors: {index_description.total_vector_count}")

    except Exception as e:
        if is_dev_mode():
            st.sidebar.error(f"Error optimizing index: {str(e)}")


# Call this after index initialization
optimize_index()


# Add this after optimize_index()
def check_index_stats():
    """
    Check and display Pinecone index statistics
    """
    try:
        stats = index.describe_index_stats()
        if is_dev_mode():
            st.sidebar.write("### Pinecone Index Stats")
            st.sidebar.write(f"Total vectors: {stats.total_vector_count}")
            st.sidebar.write(f"Namespaces: {list(stats.namespaces.keys())}")
        return stats.total_vector_count > 0
    except Exception as e:
        if is_dev_mode():
            st.sidebar.error(f"Error checking index stats: {str(e)}")
        return False


# Add these before the chat history initialization
if not check_index_stats():
    st.warning(
        "⚠️ The Pinecone index appears to be empty. Please add some data before querying."
    )


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_embedding(text):
    """
    Cache embeddings to avoid redundant API calls
    """
    response = client.embeddings.create(
        model="text-embedding-3-small", input=text, encoding_format="float"
    )
    return response.data[0].embedding


@st.cache_data(ttl=3600)
def get_context_from_pinecone(query):
    """
    Get relevant context from Pinecone based on the query
    """
    # Get cached embedding for query
    query_vector = get_embedding(query)

    # Use hybrid search for better results
    results = index.query(
        vector=query_vector,
        namespace=os.getenv("PINECONE_NAMESPACE"),
        top_k=15,
        include_metadata=True,
        include_values=False,  # Don't return vector values to reduce payload
        sparse_vector=None,  # Optional: Add sparse vector for hybrid search
    )

    contexts = []
    seen_texts = set()  # Track unique texts to avoid duplicates

    for match in results.matches:
        if match.score > 0.5:
            text = match.metadata.get("text", "")
            # Deduplicate similar content
            text_clean = " ".join(text.replace("\xa0", " ").split())
            if text_clean not in seen_texts:
                seen_texts.add(text_clean)
                contexts.append(text_clean)

    return "\n\n".join(contexts)


def get_chatgpt_response(messages, context=""):
    """
    Get response from ChatGPT with context
    """
    # Create a more detailed system message
    system_message = {
        "role": "system",
        "content": """You are a knowledge assistant for the Town of Greenburgh. Your role is to provide detailed, well-structured answers using the provided context.

        When answering:
        1. Use most recent information and friendly language when responding  
        2. Provide relevant details and explanations
        3. If applicable, mention specific rules or regulations
        4. If there are exceptions or special conditions, clearly state them
        5. Use bullet points or numbered lists when appropriate to improve readability
        6. Provide links to relevant sections of the Greenburgh website when possible

        Only use information from the following context. If you cannot answer completely based on the context, acknowledge what you can answer and what information is missing.

        Context: {context}""".format(
            context=(
                context
                if context
                else "Could not locate information. Please check the official website."
            )
        ),
    }

    # Always include system message at the start
    messages = [system_message] + messages

    stream = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        stream=True,
        temperature=0.7,  # Increased for more natural responses
    )
    return stream


def validate_context_relevance(query, context):
    """
    Validate if the context is relevant to the query
    """
    if not context:
        return "Sorry, I was unable to locate that information. You might want to check the official Greenburgh website for the most up-to-date details."

    # Get similarity score between query and context
    response = client.embeddings.create(
        model="text-embedding-3-small", input=[query, context], encoding_format="float"
    )

    query_embedding = response.data[0].embedding
    context_embedding = response.data[1].embedding

    similarity = cosine_similarity(query_embedding, context_embedding)

    if similarity < 0.5:
        return "Sorry, I was unable to locate that specific information. Would you like to try rephrasing your question?"

    return None


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


# Update the markdown styling with Greenburgh's colors
st.markdown(
    """
    <style>
        /* Greenburgh theme colors */
        :root {
            --greenburgh-green: #004831;
            --greenburgh-light: #ffffff;
            --greenburgh-accent: #b8860b;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: var(--greenburgh-green) !important;
        }
        
        /* Chat container styling */
        .stChatMessage {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        /* Input box styling */
        .stTextInput > div > div > input {
            border-color: var(--greenburgh-green);
        }
        
        /* Button styling */
        .stButton > button {
            background-color: var(--greenburgh-green);
            color: white;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Update the welcome header with new branding
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #004831; margin-bottom: 0;'>🧞‍♂️ Welcome to Greenburgh Genie</h1>
        <h3 style='color: #004831; font-weight: normal; margin-top: 10px;'>Your AI-Powered Search Assistant</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Update the welcome message with new content
if "messages" not in st.session_state:
    # Welcome text
    st.markdown(
        """
        <div style='padding: 25px; border-radius: 10px; background-color: #f8f9f9; border: 2px solid #004831;'>
            <p style='font-size: 16px; margin-bottom: 20px; '>
                Just type your question, and I'll fetch the answers from www.greenburghny.com for you—fast, easy, and hassle-free!
            </p>
            Try Asking:
            <ul style='font-size: 14px; font-weight: light; font-style: italic;'>
                <li>What are Greenburgh's regulations for usage of gas leaf blowers?</li>
                <li>How do I dispose off old paint cans?</li>
                <li>Do I need permits to trim trees on my property?</li>
                <li>What are the town's regulations for constructing a fence?</li>
                <li>What are electricity rates in Greenburgh?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.messages = []

# Initialize chat history and performance metrics
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize performance metrics
if "query_times" not in st.session_state:
    st.session_state.query_times = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Move this function definition up, before the chat input section
def monitor_performance():
    """
    Monitor and display performance metrics
    """
    if is_dev_mode():
        # Display metrics in sidebar
        st.sidebar.write("### Performance Metrics")
        if st.session_state.query_times:
            avg_time = sum(st.session_state.query_times) / len(
                st.session_state.query_times
            )
            st.sidebar.write(f"Avg Query Time: {avg_time:.2f}s")
        st.sidebar.write(f"Total Queries: {st.session_state.total_queries}")


# Update the get_client_ip function
def get_client_ip():
    """Get client IP address from Streamlit"""
    try:
        # Get the request headers from Streamlit's runtime
        headers = st.runtime.get_instance()._session.ws.request.headers

        # Try to get IP from X-Forwarded-For header first (for proxied requests)
        forwarded_for = headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Fallback to direct remote address
        remote_addr = headers.get("Remote-Addr")
        if remote_addr:
            return remote_addr

        return "Unknown"
    except:
        return "Unknown"


# Then keep the chat input section
if prompt := st.chat_input("What would you like to know?"):
    start_time = time.time()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a friendly loading message
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching for relevant information..."):
            # Get relevant context from Pinecone
            context = get_context_from_pinecone(prompt)

            # Validate context relevance
            validation_message = validate_context_relevance(prompt, context)

        if validation_message:
            # Log query with context_found=False
            query_id = log_query(prompt, context_found=False, client_ip=get_client_ip())
            st.write(validation_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": validation_message}
            )
        else:
            # Log query with context_found=True
            query_id = log_query(prompt, context_found=True, client_ip=get_client_ip())

            # Continue with normal flow
            stream = get_chatgpt_response(
                [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                context,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

    query_time = time.time() - start_time
    st.session_state.query_times.append(query_time)
    st.session_state.total_queries += 1
    monitor_performance()


# Update the disclaimer in sidebar with more accurate description and feedback link
st.sidebar.markdown(
    """
    ### About Greenburgh Genie
    <div style='font-size: 14px; color: #666; margin-bottom: 20px;'>
        This is an experimental hobby project and is not affiliated with the official Greenburgh government website. 
        While we strive to provide accurate information, always verify with 
        <a href='https://www.greenburghny.com' target='_blank' style='color: #004831;'>official sources</a> when needed.
    </div>
    
    ### Feedback & Support
    <div style='font-size: 14px; color: #666;'>
        Have feedback or concerns? Email us at:
        <a href='mailto:greenburgh_genie@proton.me' style='color: #004831;'>greenburgh_genie@proton.me</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep the injection call at the start of the app
inject_ga()


def find_and_delete_vectors(search_text):
    """
    Find and delete vectors containing specific text (dev mode only)
    """
    if not is_dev_mode():
        return "This function is only available in development mode"

    # Get the embedding for the search text
    search_vector = get_embedding(search_text)

    # Search for similar vectors
    results = index.query(
        vector=search_vector,
        namespace=os.getenv("PINECONE_NAMESPACE"),
        top_k=5,
        include_metadata=True,
    )

    # Return the matches for review
    return results.matches


def delete_vector(vector_id):
    """Delete a specific vector by ID"""
    if not is_dev_mode():
        return

    index.delete(ids=[vector_id], namespace=os.getenv("PINECONE_NAMESPACE"))


# Add this after the monitor_performance section
if is_dev_mode():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Database Management")

    search_text = st.sidebar.text_input(
        "Search for content to remove:", placeholder="Enter text to find in database"
    )

    if search_text:
        matches = find_and_delete_vectors(search_text)
        if matches:
            st.sidebar.markdown("#### Found Entries:")
            for match in matches:
                with st.sidebar.expander(f"Score: {match.score:.2f}", expanded=True):
                    st.write(match.metadata.get("text", "No text found"))
                    if st.button("Delete this entry", key=match.id):
                        delete_vector(match.id)
                        st.success(
                            "Entry deleted! Refresh the page to see updated results."
                        )
        else:
            st.sidebar.info("No matching entries found")
