from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434/v1"
    model_name: str = "qwen-agent"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
