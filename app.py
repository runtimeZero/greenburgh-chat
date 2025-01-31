import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime
import time

# Must be the first Streamlit command
st.set_page_config(page_title="Greenburgh Guide", page_icon="🏛️", layout="wide")

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

        st.sidebar.write("### Index Configuration")
        st.sidebar.write(f"Pod Type: {index_config['pod_type']}")
        st.sidebar.write(f"Vectors: {index_description.total_vector_count}")

    except Exception as e:
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
        st.sidebar.write("### Pinecone Index Stats")
        st.sidebar.write(f"Total vectors: {stats.total_vector_count}")
        st.sidebar.write(f"Namespaces: {list(stats.namespaces.keys())}")
        return stats.total_vector_count > 0
    except Exception as e:
        st.sidebar.error(f"Error checking index stats: {str(e)}")
        return False


# Add these before the chat history initialization
if not check_index_stats():
    st.warning(
        "⚠️ The Pinecone index appears to be empty. Please add some data before querying."
    )


# Add test data button in sidebar
if st.sidebar.button("Add Test Data"):
    add_test_data()


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
        top_k=5,
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
        "content": """You are a knowledgeable assistant for the Town of Greenburgh. Your role is to provide detailed, well-structured answers using the provided context.

        When answering:
        1. Start with a clear, direct response to the question
        2. Provide relevant details and explanations
        3. If applicable, mention specific rules or regulations
        4. If there are exceptions or special conditions, clearly state them
        5. Use bullet points or numbered lists when appropriate to improve readability

        Only use information from the following context. If you cannot answer completely based on the context, acknowledge what you can answer and what information is missing.

        Context: {context}""".format(
            context=context if context else "No context available."
        ),
    }

    # Always include system message at the start
    messages = [system_message] + messages

    stream = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        stream=True,
        temperature=0.3,  # Slightly increased for more natural responses while maintaining accuracy
    )
    return stream


def validate_context_relevance(query, context):
    """
    Validate if the context is relevant to the query
    """
    if not context:
        return "Sorry, no relevant information found."

    # Get similarity score between query and context
    response = client.embeddings.create(
        model="text-embedding-3-small", input=[query, context], encoding_format="float"
    )

    query_embedding = response.data[0].embedding
    context_embedding = response.data[1].embedding

    similarity = cosine_similarity(query_embedding, context_embedding)

    # Lower this threshold as well
    if similarity < 0.5:  # Changed from 0.7
        return "I found some information, but it may not be directly relevant to your query."

    return None


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def add_test_data():
    """
    Add test data to Pinecone with batch processing
    """
    BATCH_SIZE = 100  # Process vectors in batches

    test_data = [
        """In Greenburgh, leaf blowers are strictly regulated under the Town's Noise Ordinance. Gas-powered blowers are prohibited from operating above 55 decibels, which is the town's established noise limit.""",
        """The Town's regulations on blowers cover three main areas: 1) Noise pollution - both gas and electric blowers must meet decibel limits, 2) Environmental impact - concerns about air-borne particulates and soil damage, 3) Health considerations - protecting residents from involuntary exposure to harmful particles.""",
        """Residents have special considerations under the local law. While using electric blowers, residents have certain exemptions, but must still comply with basic noise and usage restrictions. These exemptions were added to make the law more acceptable to homeowners.""",
        """Modern electric blowers, despite being marketed as environmentally friendly, can produce Hurricane-force winds and high decibel levels. Almost all blowers, both gas and electric, operate at decibel levels above EPA comfort standards. Out of 300 studied blowers, only 5 electric models met the Town's noise requirements.""",
        """The Town's comprehensive approach to blower regulation addresses: a) involuntary exposure of residents to harmful air-borne particulates, b) noise pollution affecting quality of life, and c) prevention of soil damage in the community.""",
        """Enforcement of blower regulations in Greenburgh focuses on both commercial operators and residential users. The regulations apply year-round but may have seasonal variations in restrictions.""",
        "The Town of Greenburgh is located in Westchester County, New York, and was founded in 1788.",
        "Greenburgh encompasses several villages including Ardsley, Dobbs Ferry, and Irvington, each subject to town-wide regulations.",
    ]

    # Get embeddings for all texts at once
    response = client.embeddings.create(
        model="text-embedding-3-small", input=test_data, encoding_format="float"
    )

    # Prepare vectors in batches
    vectors = []
    for i, embedding in enumerate(response.data):
        vectors.append(
            {
                "id": f"test-{i}",
                "values": embedding.embedding,
                "metadata": {
                    "text": test_data[i],
                    "timestamp": str(datetime.now()),  # Add timestamp for versioning
                    "source": "test_data",
                },
            }
        )

    # Upsert vectors in batches
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)

    st.success(f"Successfully added {len(vectors)} vectors to Pinecone")


# Where the old set_page_config was, keep only the markdown styling
st.markdown(
    """
    <h1 style='text-align: center; color: #2E4053;'>🏛️ Your Greenburgh Guide</h1>
    <h3 style='text-align: center; color: #566573;'>Your AI-powered assistant for all things Greenburgh</h3>
    """,
    unsafe_allow_html=True,
)

# Add a welcoming message
if "messages" not in st.session_state:
    st.markdown(
        """
    <div style='padding: 20px; border-radius: 10px; background-color: #F8F9F9;'>
        👋 <b>Welcome to Your Greenburgh Guide!</b>
        <br><br>
        I'm here to help you with:
        <br>
        • Town regulations and policies 📋
        <br>
        • Local services and facilities 🏢
        <br>
        • Community information 🏘️
        <br><br>
        How can I assist you today?
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
    # Display metrics in sidebar
    st.sidebar.write("### Performance Metrics")
    if st.session_state.query_times:
        avg_time = sum(st.session_state.query_times) / len(st.session_state.query_times)
        st.sidebar.write(f"Avg Query Time: {avg_time:.2f}s")
    st.sidebar.write(f"Total Queries: {st.session_state.total_queries}")


# Then keep the chat input section
if prompt := st.chat_input("What would you like to know?"):
    start_time = time.time()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get relevant context from Pinecone
    context = get_context_from_pinecone(prompt)

    # Validate context relevance
    validation_message = validate_context_relevance(prompt, context)
    if validation_message:
        with st.chat_message("assistant"):
            st.write(validation_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": validation_message}
        )
    else:
        # Continue with normal flow
        with st.chat_message("assistant"):
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
