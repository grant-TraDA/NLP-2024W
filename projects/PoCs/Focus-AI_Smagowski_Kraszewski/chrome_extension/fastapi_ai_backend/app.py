from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import logging
import html2text
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    CRITIC_LLM: str
    OPENAI_API_KEY: str
    model_config = SettingsConfigDict(
            env_file='.env',
            env_file_encoding='utf-8',
            case_sensitive=True
        )

settings = Settings()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize OpenAI client
logger.info("Initializing Langchain OpenAI client")
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    api_key=settings.OPENAI_API_KEY
)

print(settings.OPENAI_API_KEY)

# Add CORS middleware configuration
logger.info("Configuring CORS middleware")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

class WebpageRequest(BaseModel):
    topic: str
    content: str

@app.post("/validate-webpage")
async def validate_webpage(request: WebpageRequest):
    logger.info(f"Received validation request for topic: {request.topic}")
    
    try:
        # Convert HTML to markdown
        h = html2text.HTML2Text()
        h.ignore_links = True
        markdown_text = h.handle(request.content)
        
        # Get first 3000 characters
        truncated_text = markdown_text[:3000]
        
        prompt = f"""As an expert educational content curator with years of experience in personalized learning:
            Your task is to determine if this content would benefit a user who wants to: '{request.topic}'

            Consider:
            1. The educational value of the content
            2. Relevance to the user's learning objective
            3. Potential for knowledge advancement
            
            Content to evaluate: {truncated_text}
            
            Based on your expertise, output only 'True' if the content supports the learning goal, or 'False' if it's a distraction."""
        
        logger.info("Sending request to OpenAI API")
        response = model.invoke(
            input=prompt
        )
        response_text = response.content.strip()
        logger.info(f"Received response from OpenAI: {response_text}")
        
        # Return False if contains FALSE, otherwise True
        return {"result": False if "FALSE" in response_text.upper() else True}
    
    except Exception as e:
        logger.error(f"Error during webpage validation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


