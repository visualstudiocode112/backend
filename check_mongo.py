"""Quick MongoDB connectivity checker.
Reads MONGO_URL and DB_NAME from .env and attempts to connect and list databases.
Usage: python check_mongo.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import PyMongoError

ROOT = Path(__file__).parent
load_dotenv(ROOT / '.env')

MONGO_URL = os.environ.get('MONGO_URL')
DB_NAME = os.environ.get('DB_NAME')

if not MONGO_URL:
    print('MONGO_URL not set in .env')
    raise SystemExit(1)

print('Attempting to connect to MongoDB...')
try:
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    # Ping
    client.admin.command('ping')
    print('Ping succeeded')
    print('Databases:')
    for db in client.list_database_names():
        print(' -', db)
    if DB_NAME in client.list_database_names():
        print(f"Target database '{DB_NAME}' exists.")
    else:
        print(f"Target database '{DB_NAME}' not found (this may be ok if empty).")
    client.close()
    print('Connection check finished successfully')
except PyMongoError as e:
    print('MongoDB connection failed:')
    print(str(e))
    raise SystemExit(2)
