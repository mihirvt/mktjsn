"""Set up logging before importing anything else"""

import sentry_sdk

from api.constants import DEPLOYMENT_MODE, ENABLE_TELEMETRY, SENTRY_DSN
from api.logging_config import ENVIRONMENT, setup_logging

# Set up logging and get the listener for cleanup
setup_logging()

import sys
import asyncio

# Patch Python 3.12 asyncio datagram transport bug (aioice fix)
if sys.version_info >= (3, 12):
    try:
        from asyncio.selector_events import _SelectorDatagramTransport
        original_sendto = _SelectorDatagramTransport.sendto

        def patched_sendto(self, data, addr=None):
            if not getattr(self, '_sock', None):
                return
            try:
                return original_sendto(self, data, addr)
            except Exception:
                if not getattr(self, '_loop', None):
                    return
                raise
        _SelectorDatagramTransport.sendto = patched_sendto
    except ImportError:
        pass


if SENTRY_DSN and (
    DEPLOYMENT_MODE != "oss" or (DEPLOYMENT_MODE == "oss" and ENABLE_TELEMETRY)
):
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        send_default_pii=True,
        environment=ENVIRONMENT,
    )
    print(f"Sentry initialized in environment: {ENVIRONMENT}")


from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes.main import router as main_router
from api.tasks.arq import get_arq_redis

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warmup arq pool
    await get_arq_redis()

    yield  # Run app

    # Shutdown sequence - this runs when FastAPI is shutting down
    logger.info("Starting graceful shutdown...")


app = FastAPI(
    title="Dograh API",
    description="API for the Dograh app",
    version="1.0.0",
    openapi_url=f"{API_PREFIX}/openapi.json",
    lifespan=lifespan,
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

api_router = APIRouter()

# include subrouters here
api_router.include_router(main_router)

# main router with api prefix
app.include_router(api_router, prefix=API_PREFIX)
