from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_BASE_URL: str = "http://localhost:11434/v1"
    MODEL_NAME: str = ""
    INDENT_MODEL_NAME: str = ""
    API_KEY: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
