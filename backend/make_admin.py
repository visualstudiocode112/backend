#!/usr/bin/env python3
"""
Script para hacer admin a un usuario
Uso: python make_admin.py email@example.com
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

async def make_admin(email: str):
    """Make a user admin by email"""
    mongo_url = os.environ['MONGO_URL']
    db_name = os.environ['DB_NAME']
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    
    try:
        # Find user by email
        user = await db.users.find_one({"email": email})
        
        if not user:
            print(f"❌ Usuario no encontrado: {email}")
            return False
        
        # Update user role to admin
        result = await db.users.update_one(
            {"email": email},
            {"$set": {"role": "admin"}}
        )
        
        if result.modified_count > 0:
            print(f"✅ Usuario {email} es ahora ADMINISTRADOR")
            return True
        else:
            print(f"⚠️  Usuario ya era administrador: {email}")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python make_admin.py email@example.com")
        sys.exit(1)
    
    email = sys.argv[1]
    success = asyncio.run(make_admin(email))
    sys.exit(0 if success else 1)
