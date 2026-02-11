import websockets
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db.base import init_db
from .core.ws_manager import WebSocketManager
from . import api


# Global WebSocket manager
ws_manager: WebSocketManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting CrowdCompute FastAPI Foreman...")

    # Initialize database with comprehensive seeding
    from foreman.db.seed import initialize_database

    await initialize_database()

    # Initialize WebSocket manager
    global ws_manager
    ws_manager = WebSocketManager()

    # Set ws_manager in api.websockets module
    api.websockets.set_ws_manager(ws_manager)

    # Start WebSocket server in background task
    async def start_websocket_server():
        try:
            # Configure WebSocket server with proper timeouts
            # ping_interval: How often to send pings to keep connection alive (None = disabled, let our custom ping handle it)
            # ping_timeout: How long to wait for pong response before closing
            # close_timeout: How long to wait for close handshake
            # max_size: Maximum message size (10MB to handle large task results)
            websocket_server = await websockets.serve(
                ws_manager.handle_connection,
                "0.0.0.0",
                9000,
                ping_interval=None,  # Disable built-in ping, we handle it ourselves
                ping_timeout=None,  # Disable ping timeout
                close_timeout=30,  # Wait up to 30s for close handshake
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            print("WebSocket server started on ws://localhost:9000")
            await websocket_server.wait_closed()
        except Exception as e:
            print(f"WebSocket server error: {e}")
            import traceback

            traceback.print_exc()

    # Start WebSocket server as background task
    import asyncio

    websocket_task = asyncio.create_task(start_websocket_server())

    print("FastAPI Foreman started!")
    print("REST API: http://localhost:8000")
    print("WebSocket: ws://localhost:9000")

    yield

    # Shutdown
    print("Shutting down FastAPI Foreman...")
    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass


# Create FastAPI app
app = FastAPI(
    title="CrowdCompute Foreman",
    description="FastAPI-based foreman server for distributed computing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes
app.include_router(api.routes.router)
app.include_router(api.websockets.router)
app.include_router(api.scheduler_routes.router)
app.include_router(api.checkpoint_routes.router)

# Include evaluation routes
from .api import evaluation_routes

app.include_router(evaluation_routes.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
