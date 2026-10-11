import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(default="StrongPassword123!", validation_alias="NEO4J_PASSWORD")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="qwen2.5:7b-instruct", validation_alias="LLM_MODEL")
    embedding_model: str = Field(default="all-minilm", validation_alias="EMBEDDING_MODEL")
    similarity_threshold: float = Field(default=0.65, validation_alias="SIMILARITY_THRESHOLD")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    embedding_cache_size: int = 2048
    llm_cache_size: int = 1024
    max_workers: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def setup_logger(name: str = "m4") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel("INFO")
    return logger
