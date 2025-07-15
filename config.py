import os
from dotenv import load_dotenv

def load_config():
    if not os.path.exists(".env"):
        raise FileNotFoundError("Missing `.env` file with API credentials")

    load_dotenv()
    required_keys = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    config = {}

    for key in required_keys:
        value = os.getenv(key)
        print("Debug", f"(Key={key}, Value = {value})")
        if not value:
            raise EnvironmentError(f"Missing key in .env: {key}")
        config[key] = value

    return config
