from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

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
