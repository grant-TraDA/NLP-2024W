import pandas as pd
from langchain_openai import ChatOpenAI
import os
import logging
from datetime import datetime
from tqdm import tqdm
from pydantic_settings import BaseSettings, SettingsConfigDict
import random
import numpy as np

# Set fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Set up logging
logging.basicConfig(
    filename='prompt_testing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PromptTester:
    def __init__(self, api_key, seed=42):
        self.api_key = api_key
        self.seed = seed
        self.models = ['gpt-4o-mini', 'gpt-3.5-turbo']
        self.prompts = {
            'chain_of_thought_few_shot': """Let's analyze this step by step:
            1. Understand the user's learning objective: '{label}'
            2. Examine the page content: {content}
            3. Identify key educational concepts in the content
            4. Check if these concepts align with the user's learning goal
            5. Consider whether this content would help achieve the learning objective
            6. Evaluate if the content is worth the user's time and attention
            
            Here are some examples of matching content to learning objectives:
            User's Goal: "Learn Python Programming"
            Content: "A comprehensive guide to Python functions, loops, and data structures..."
            Answer: True (Content directly supports learning objective)

            User's Goal: "Study World War II"
            Content: "Top 10 recipes for chocolate chip cookies..."
            Answer: False (Content would distract from learning objective)
            
            Based on this analysis, output only 'True' if the content is relevant to the user's learning goal, or 'False' if it would be a distraction.""",
            
            'few_shot_learning': """Here are some examples of matching content to learning objectives:
            User's Goal: "Learn Python Programming"
            Content: "A comprehensive guide to Python functions, loops, and data structures..."
            Answer: True (Content directly supports learning objective)

            User's Goal: "Study World War II"
            Content: "Top 10 recipes for chocolate chip cookies..."
            Answer: False (Content would distract from learning objective)

            Now, evaluate this case:
            User wants to: '{label}'
            Content: {content}
            Output only 'True' or 'False'.""",
            
            'role_prompting': """As an expert educational content curator with years of experience in personalized learning:
            Your task is to determine if this content would benefit a user who wants to: '{label}'

            Consider:
            1. The educational value of the content
            2. Relevance to the user's learning objective
            3. Potential for knowledge advancement
            
            Content to evaluate: {content}
            
            Based on your expertise, output only 'True' if the content supports the learning goal, or 'False' if it's a distraction.""",
            
            'zero_shot_chain_of_thought': """Let's evaluate the learning opportunity:
            1. What is the user trying to learn or achieve? ('{label}')
            2. What knowledge does this content provide?
            3. Would engaging with this content advance the user's goal?
            4. Could this content distract from the learning objective?
            5. Is this the right time to engage with this content?

            After considering these points, output only 'True' if the user should engage with this content, or 'False' if they should skip it.""",
            
            'structured_reasoning': """Follow this learning assessment framework:
            1. User Objective Analysis:
               - Primary learning goal: '{label}'
               - Current focus requirement
            2. Content Evaluation:
               - Knowledge contribution
               - Relevance to objective
            3. Decision Factors:
               - Direct benefit to learning
               - Potential for distraction
               - Time-value assessment
               
            Content to analyze: {content}
            
            Based on this framework, output only 'True' if the content aligns with the learning objective, or 'False' if it should be blocked."""
        }

    def create_output_directory(self):
        """Create output directory for results if it doesn't exist"""
        os.makedirs('prompt_results', exist_ok=True)

    def validate_and_extract_response(self, response_text):
        """Validate and extract True/False from response"""
        response_upper = response_text.upper()
        if 'TRUE' in response_upper:
            return 'TRUE'
        elif 'FALSE' in response_upper:
            return 'FALSE'
        return None

    def get_model_response(self, prompt, model, max_retries=5):
        """Get response from OpenAI model with validation and retry logic"""
        for attempt in range(max_retries):
            try:
                model = ChatOpenAI(
                    temperature=0.0,
                    model_name=model,
                    api_key=self.api_key,
                    seed=self.seed
                )
                response = model.invoke(
                    input=prompt,
                    temperature=0
                )

                response_text = response.content.strip()
                validated_response = self.validate_and_extract_response(response_text)
                
                if validated_response:
                    return validated_response
                
                logger.warning(f"Invalid response format: {response_text}. Retrying...")
                continue
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for model {model}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        raise ValueError("Failed to get valid True/False response after all retries")

    def run_prompt_testing(self, df):
        """Run testing of all prompts with all models on the dataset"""
        self.create_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []

        for model in self.models:
            logger.info(f"Starting testing with model: {model}")
            
            for prompt_name, prompt_template in self.prompts.items():
                for _, row in tqdm(df.iterrows(), total=len(df), 
                                 desc=f"Processing with {model} - {prompt_name}"):
                    prompt = prompt_template.format(
                        content=row['markdown_content'],
                        label=row['matching_label']
                    )
                    
                    try:
                        response = self.get_model_response(prompt, model)
                        all_results.append({
                            'content': row['markdown_content'],
                            'true_label': row['matching_status'],
                            'predicted_label': response == 'TRUE',
                            'model': model,
                            'prompt_method': prompt_name
                        })
                        logger.info(f"Successfully processed row with {model} and {prompt_name}")
                    except Exception as e:
                        logger.error(f"Error processing row: {str(e)}")
                        continue

                # Log completion
                logger.info(f"Completed testing for {model} with {prompt_name}")
        
        # Save all results to a single file
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'prompt_results/all_results_{timestamp}.csv', index=False)

# Usage example
if __name__ == "__main__":
    class Settings(BaseSettings):
        CRITIC_LLM: str
        OPENAI_API_KEY: str
        model_config = SettingsConfigDict(
            env_file='.env',
            env_file_encoding='utf-8',
            case_sensitive=True
        )
    
    settings = Settings()
    # Load your dataset
    df = pd.read_csv('processed-data/ready_dataset.csv')
    
    # Initialize the tester with your API key
    tester = PromptTester(settings.OPENAI_API_KEY)
    
    # Run the testing
    tester.run_prompt_testing(df)
    
    print("Testing completed. Check the 'prompt_results' folder for detailed results.")
