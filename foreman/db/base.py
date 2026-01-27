from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from contextlib import asynccontextmanager

# Database URL
DATABASE_URL = "sqlite+aiosqlite:///./crowdcompute.db"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Alias for convenience
async_session = AsyncSessionLocal

# Base class for models
Base = declarative_base()


# Database functions
async def get_db():
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session


# Alias for convenience
get_db_session = db_session


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
