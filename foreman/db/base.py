from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.exc import OperationalError
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import os
import sqlite3

# Database URL
DATABASE_URL = "sqlite+aiosqlite:///./crowdio.db"

# Create async engine
# timeout (seconds) lets SQLite wait for transient locks instead of failing fast.
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"timeout": 30},
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Alias for convenience
async_session = AsyncSessionLocal

# Base class for models
Base = declarative_base()


def _sqlite_file_path() -> str:
    """Resolve filesystem path for the configured SQLite database."""
    prefix = "sqlite+aiosqlite:///"
    if not DATABASE_URL.startswith(prefix):
        return ""

    raw = DATABASE_URL[len(prefix):]
    if raw.startswith("./"):
        raw = os.path.join(os.getcwd(), raw[2:])
    return os.path.abspath(raw)


def _quarantine_corrupt_db(db_path: str, reason: str) -> str:
    """Move a malformed DB to a timestamped backup path and return destination."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.corrupt_{ts}.bak"
    os.replace(db_path, backup_path)

    # SQLite sidecar files can keep stale state; remove if present.
    for sidecar in ("-wal", "-shm"):
        sidecar_path = db_path + sidecar
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

    print(
        f"⚠️  Corrupt SQLite DB detected ({reason}). "
        f"Moved to: {backup_path}"
    )
    print("🛠️  A fresh database will be created automatically.")
    return backup_path


def ensure_sqlite_integrity() -> bool:
    """
    Verify SQLite integrity and auto-recover from malformed files.

    Returns:
        bool: True if DB is usable after check/recovery.
    """
    db_path = _sqlite_file_path()
    if not db_path:
        return True

    if not os.path.exists(db_path):
        return True

    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
        finally:
            conn.close()

        msg = row[0] if row else "unknown"
        if msg == "ok":
            return True

        _quarantine_corrupt_db(db_path, msg)
        return True

    except sqlite3.DatabaseError as exc:
        _quarantine_corrupt_db(db_path, str(exc))
        return True
    except Exception as exc:
        print(f"❌ Failed to run SQLite integrity check: {exc}")
        return False


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
    if not ensure_sqlite_integrity():
        raise RuntimeError("Database integrity check failed")

    # Retry startup DDL if another process is briefly holding a lock
    # (e.g., editor extension, parallel startup, stale writer transaction).
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            async with engine.begin() as conn:
                await conn.exec_driver_sql("PRAGMA busy_timeout = 30000")
                await conn.exec_driver_sql("PRAGMA journal_mode = WAL")
                await conn.run_sync(Base.metadata.create_all)
            return
        except OperationalError as exc:
            msg = str(exc).lower()
            if "database is locked" not in msg:
                raise

            if attempt == max_retries:
                raise RuntimeError(
                    "SQLite database is locked and could not be initialized after retries. "
                    "Close any DB viewers/writers holding crowdio.db and retry."
                ) from exc

            wait_seconds = attempt
            print(
                f"⚠️  SQLite locked during init (attempt {attempt}/{max_retries}). "
                f"Retrying in {wait_seconds}s..."
            )
            await asyncio.sleep(wait_seconds)
