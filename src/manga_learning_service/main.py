from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from manga_learning_service.ai.routes import router as ai_router
from manga_learning_service.anki.routes import router as anki_router
from manga_learning_service.cache.store import CacheStore
from manga_learning_service.config import Settings, get_settings
from manga_learning_service.middleware.cors import install_cors
from manga_learning_service.ocr.routes import router as ocr_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    cache = CacheStore(settings.cache_db_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await cache.init()
        _app.state.cache = cache
        try:
            yield
        finally:
            await cache.close()

    app = FastAPI(title="manga-learning-service", version="0.1.0", lifespan=lifespan)
    install_cors(app, settings.cors_allow_origins)
    app.include_router(ocr_router, prefix="/ocr")
    app.include_router(anki_router, prefix="/anki")
    app.include_router(ai_router, prefix="/ai")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _build_default_app() -> FastAPI:
    return create_app()


app = _build_default_app()
