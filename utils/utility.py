import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Load env variable from config/.env
env_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
env_path = os.path.abspath(env_path)
load_dotenv(env_path)

print(f'Loaded the env file from this location: {env_path}')

def get_embedding_model(provider: str):
    provider = provider.lower()

    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPEN_API_KEY not found')
        print("Open API key loaded!!")
        return OpenAIEmbeddings(api_key = api_key)
    elif provider in ['gemini', 'google']:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError('GEMINI_API_KEY not found')
        print("Gemini API key loaded!!")
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
    else:
        raise ValueError(f'Unsupported embedding provider: {provider}')

