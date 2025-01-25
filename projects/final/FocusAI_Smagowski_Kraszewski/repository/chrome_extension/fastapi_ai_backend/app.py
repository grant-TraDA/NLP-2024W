from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import logging
import html2text
from pydantic_settings import BaseSettings, SettingsConfigDict
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage
import os

class LLMWorkflow:
    def call_model(self, state: MessagesState):
        """Call the model with the current state."""
        messages = state["messages"]

        response = self.model.invoke(messages)
        return {"messages": [response]}

    def get_workflow(self):
        # Create workflow graph
        workflow = StateGraph(MessagesState)

        # Define nodes
        workflow.add_node("agent", self.call_model)

        # Set entry point
        workflow.add_edge(START, "agent")

        workflow.add_edge("agent", END)

        return workflow

    def __init__(self, settings):
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        self.settings = settings
        
        self.model = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY
        )

        self.workflow = self.get_workflow()
        self.app = self.workflow.compile()
    
    def run(self, message: str):
        """Run the workflow with a message"""
        try:
            last_event = None
            for event in self.app.stream(
                {"messages": [HumanMessage(content=message)]},
                stream_mode="values"
            ):
                event["messages"][-1].pretty_print()
                last_event = event

            return last_event["messages"][-1].content if last_event else None
        except Exception as e:
            print(f"Error in arun: {str(e)}")
            raise


class Settings(BaseSettings):
    OPENAI_MODEL: str
    OPENAI_API_KEY: str
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_PROJECT: str
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

llm_workflow = LLMWorkflow(settings)


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
        
        prompt = f"""Let's evaluate the learning opportunity:
            1. What is the user trying to learn or achieve? ('{request.topic}')
            2. What knowledge does this content provide?
            3. Would engaging with this content advance the user's goal?
            4. Could this content distract from the learning objective?
            5. Is this the right time to engage with this content?

            If the content is a privacy information, output 'True'.
            After considering these points, output only 'True' if the user should engage with this content, or 'False' if they should skip it.
            
            Content to analyze: {truncated_text}"""
        
        logger.info("Sending request to OpenAI API")
        response_text = llm_workflow.run(prompt)
        logger.info(f"Received response from OpenAI: {response_text}")
        
        # Return False if contains FALSE, otherwise True
        return {"result": False if "FALSE" in response_text.upper() else True}
    
    except Exception as e:
        logger.error(f"Error during webpage validation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


