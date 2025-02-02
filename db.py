from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId

load_dotenv()


def get_mongo_client():
    """Get MongoDB client using connection string from environment variables"""
    mongo_uri = os.getenv("MONGODB_URI")
    return MongoClient(mongo_uri)


def log_query(query, context_found=True, response=None):
    """Log user query to MongoDB"""
    try:
        client = get_mongo_client()
        db = client.greenburgh
        collection = db.queries

        # Create document with environment info
        doc = {
            "query": query,
            "timestamp": datetime.utcnow(),
            "context_found": context_found,
            "environment": os.getenv("ENV", "dev"),
        }

        # Insert document and return its ID
        result = collection.insert_one(doc)
        client.close()
        return str(result.inserted_id)

    except Exception as e:
        print(f"Error logging query to MongoDB: {str(e)}")
        return None
