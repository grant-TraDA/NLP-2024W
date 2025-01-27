import os
from typing import List, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

project_dir = os.getcwd()
cache_dir = os.path.join(project_dir, ".cache")

model_kwargs = {"torch_dtype": torch.bfloat16}

# MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
# MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MODEL_NAME = "jinaai/jina-embeddings-v3"

model = SentenceTransformer(
    MODEL_NAME,
    cache_folder=cache_dir,
    trust_remote_code=True,
    model_kwargs=model_kwargs,
)

app = FastAPI()


class MessageRequest(BaseModel):
    messages: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


@app.post("/embeddings", response_model=EmbeddingResponse)
def get_embeddings(request: MessageRequest):
    messages = (
        request.messages if isinstance(request.messages, list) else [request.messages]
    )

    if not messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    embeddings = model.encode(messages).tolist()

    return EmbeddingResponse(embeddings=embeddings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
