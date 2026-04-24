"""
本地 OpenAI 兼容 Embedding 服务
================================
将 sentence-transformers/all-MiniLM-L6-v2 包装为 OpenAI /v1/embeddings 接口，
供 GraphRAG 官方基线使用（因为 Kimi API 不提供 embedding 接口）。

启动方式：
    python local_embedding_server.py
    # 默认监听 http://localhost:11434

GraphRAG settings.yaml 中配置：
    embedding_models:
      default_embedding_model:
        model_provider: openai
        model: all-MiniLM-L6-v2
        api_key: local
        api_base: http://localhost:11434/v1
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 模型加载（启动时一次性加载）──────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
logger.info(f"加载 embedding 模型：{MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
logger.info("模型加载完成")

# ── FastAPI 应用 ──────────────────────────────────────────────────────────────
app = FastAPI(title="Local Embedding Server", version="1.0.0")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "all-MiniLM-L6-v2"
    encoding_format: Optional[str] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    t0 = time.time()

    # 统一为列表
    texts: List[str] = request.input if isinstance(request.input, list) else [request.input]

    # 编码（L2 归一化，与 Naive RAG 索引保持一致）
    embeddings: np.ndarray = _model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    data = [
        EmbeddingData(index=i, embedding=emb.tolist())
        for i, emb in enumerate(embeddings)
    ]

    # 粗略估算 token 数（每个词约 1.3 token）
    total_chars = sum(len(t) for t in texts)
    approx_tokens = max(1, total_chars // 4)

    elapsed = time.time() - t0
    logger.info(f"  编码 {len(texts)} 条文本，耗时 {elapsed:.3f}s")

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=approx_tokens, total_tokens=approx_tokens),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434, log_level="info")
