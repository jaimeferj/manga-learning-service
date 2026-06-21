from __future__ import annotations

import logging

import uvicorn

from manga_learning_service.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [manga-learning] %(levelname)s %(message)s",
    )
    settings = get_settings()
    uvicorn.run(
        "manga_learning_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
