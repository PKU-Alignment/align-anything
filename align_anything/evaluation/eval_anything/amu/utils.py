import os
import json
import shutil
import hashlib
from datetime import datetime

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def generate_hash_uid(to_hash: dict | tuple | list | str) -> str:
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    return hash_object.hexdigest()

def copy_file(source_path, destination_path):
    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file does not exist: {source_path}") 
        if os.path.exists(destination_path):
            raise FileExistsError(f"Destination file already exists: {destination_path}")
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)
    
    except Exception as e:
        print(f"Error copying file: {str(e)}")