import os
import json
from typing import List, Dict, Tuple
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from dataclasses import dataclass
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
import pandas as pd
import uuid
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

@dataclass
class MarkdownState:
    """State for markdown processing workflow"""
    messages: List
    current_markdown: str
    matching_labels: List[str] = None
    non_matching_labels: List[str] = None

class JsonValidationInput(BaseModel):
    json_str: str

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MarkdownLabeler(metaclass=SingletonMeta):
    def __init__(self, settings):
        if not hasattr(self, '_initialized'):
            if settings is None:
                raise ValueError("Settings are required for initialization")
            
            self.settings = settings
            self.model = ChatOpenAI(
                temperature=0.0,
                model_name=settings.CRITIC_LLM,
                api_key=settings.OPENAI_API_KEY,
                seed=42  # Added seed for reproducibility
            )
            
            self.prompt_template = """You are a content labeling expert. Given the following markdown content, 
            your task is to:
            1. Generate 5-10 relevant labels that accurately describe the content. Part of them should be more general.
            2. Generate 5-10 labels that definitely DO NOT match this content.
            
            You MUST use the json_validator tool to validate your response before sending it.
            
            Format your response as a JSON string with two lists:
            {
                "matching_labels": ["label1", "label2", ...],
                "non_matching_labels": ["label1", "label2", ...]
            }

            Markdown content:
            """
            
    def call_model(self, markdown: str):
        """Process with LLM and generate labels"""
        response = self.model.invoke(input=self.prompt_template + markdown)
        return response

    
def validate_json(json_str: str) -> Dict:
    """Tool to validate JSON and check for required structure"""
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return {"valid": False, "error": "Response must be a JSON object"}
            
        # Check that there are exactly 2 fields
        if len(data.keys()) != 2:
            return {"valid": False, "error": "JSON must contain exactly 2 fields"}
            
        # Check that the fields are matching_labels and non_matching_labels
        if set(data.keys()) != {"matching_labels", "non_matching_labels"}:
            return {"valid": False, "error": "JSON must contain only matching_labels and non_matching_labels fields"}
            
        # Check that both fields are lists
        if not isinstance(data["matching_labels"], list) or not isinstance(data["non_matching_labels"], list):
            return {"valid": False, "error": "Both fields must be lists"}
            
        return {"valid": True, "data": data}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}
    
def process_dataframe(df_path: str, sample_size: int = 100) -> pd.DataFrame:
    """Process dataframe and keep only markdown_content column"""
    # Read the dataframe
    df = pd.read_csv(df_path)
    
    # Sample random rows
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Keep markdown_content column and initialize new columns
    sampled_df = sampled_df[['markdown_content']].copy()
    sampled_df['matching_labels'] = None
    sampled_df['non_matching_labels'] = None
    
    # Limit markdown_content to 3000 characters
    sampled_df['markdown_content'] = sampled_df['markdown_content'].str[:3000]
    
    return sampled_df
    

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize settings (you'll need to define this class based on your needs)
    class Settings(BaseSettings):
        CRITIC_LLM: str
        OPENAI_API_KEY: str
        model_config = SettingsConfigDict(
            env_file='.env',
            env_file_encoding='utf-8',
            case_sensitive=True
        )

    settings = Settings()
    logger.info("Initialized settings")
    
    # Initialize labeler
    labeler = MarkdownLabeler(settings)
    logger.info("Initialized MarkdownLabeler")
    
    # Process dataframe
    df = process_dataframe('processed-data/processed-files.csv')
    logger.info(f"Loaded and processed dataframe with {len(df)} rows")
    
    # Process each row
    successful_rows = 0
    failed_rows = 0
    
    for idx, row in df.iterrows():
        logger.debug(f"Processing row {idx}")
        response = labeler.call_model(row['markdown_content']).content
        # Extract JSON string between first { and last }
        json_str = response[response.find('{'):response.rfind('}')+1]
        # Validate JSON and get data
        validation_result = validate_json(json_str)
        if validation_result["valid"]:
            data = validation_result["data"]
            # Use loc instead of at for assignment
            df.loc[idx, 'matching_labels'] = str(data["matching_labels"])  # Convert list to string
            df.loc[idx, 'non_matching_labels'] = str(data["non_matching_labels"])  # Convert list to string
            successful_rows += 1
            logger.debug(f"Successfully processed row {idx}")
        else:
            failed_rows += 1
            logger.error(f"Invalid JSON for row {idx}: {validation_result['error']}")
    
    logger.info(f"Processing complete. Successful rows: {successful_rows}, Failed rows: {failed_rows}")
    
    # Save results
    df.to_csv('processed-data/labeled_files.csv', index=False)
    logger.info("Results saved to processed-data/labeled_files.csv")

if __name__ == "__main__":
    main()