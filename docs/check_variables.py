import os

# Check if the environment variable is set
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    print(f"GOOGLE_API_KEY is set: {api_key}")
else:
    print("GOOGLE_API_KEY is not set")