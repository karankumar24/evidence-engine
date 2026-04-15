"""FastAPI application factory."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evidenceengine.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="EvidenceEngine",
        version="0.1.0",
        description="Provenance-aware document verification system",
    )

    # CORS — allow all origins for dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from evidenceengine.api.routes.packets import router as packets_router
    from evidenceengine.api.routes.extraction import router as extraction_router

    app.include_router(packets_router)
    app.include_router(extraction_router)

    @app.on_event("startup")
    async def startup_event() -> None:
        """Create upload directory on startup."""
        os.makedirs(settings.upload_dir, exist_ok=True)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        return {"status": "ok"}

    return app
