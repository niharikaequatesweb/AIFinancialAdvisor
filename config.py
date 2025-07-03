import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env file manually
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

class Settings(BaseSettings):
    summarizer_model: str = "google/pegasus-xsum"#Field(default="google/pegasus-xsum", env="SUMMARIZER_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # Tokens / keys
    hf_token: str =os.environ.get("HF_API_TOKEN")
    serpapi_key: str = os.environ.get("SERP_API_KEY")

    # Chunking size
    summarizer_chunk_words: int = Field(default=50, env="SUMMARIZER_CHUNK_WORDS")

    # Vector store / cache
    vector_db_path: Path = Field(default=Path("vector.index"), env="VECTOR_DB_PATH")
    cache_path: Path = Field(default=Path("answer_cache.pkl"), env="CACHE_PATH")
    similarity_threshold: float = Field(default=0.8, env="SIMILARITY_THRESHOLD")

    class Config:
        env_file = dotenv_path  # Specify the .env file location
        extra="allow"

# Initialize settings
settings = Settings()