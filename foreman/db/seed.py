"""
Database Initialization and Seeding

This module handles first-time database setup and ensures all necessary
initial data is present.
"""

from foreman.db.base import init_db, async_session
from foreman.db.models import WorkerModel
from sqlalchemy import select


async def verify_database_schema():
    """
    Verify that all required tables exist in the database.

    Returns:
        bool: True if schema is valid, False otherwise
    """
    try:
        async with async_session() as session:
            # Try to query each critical table
            await session.execute(select(WorkerModel).limit(1))
        return True
    except Exception as e:
        print(f"❌ Database schema verification failed: {e}")
        return False


async def seed_initial_data(force: bool = False):
    """
    Seed the database with initial required data.

    Args:
        force: If True, recreate all initial data even if it exists
    """
    print("\n" + "=" * 60)
    print("🌱 Database Initialization & Seeding")
    print("=" * 60)

    # Nothing to seed yet. Add default worker templates, system settings, etc. here.
    print("No initial data required")

    print("=" * 60 + "\n")

    return 0


async def initialize_database(force_seed: bool = False):
    """
    Complete database initialization routine.

    This function:
    1. Creates all database tables (if they don't exist)
    2. Verifies the schema
    3. Seeds initial data

    Args:
        force_seed: If True, recreate initial data even if it exists
    """
    print("\n🗄️  Initializing Database...")

    # Step 1: Create tables
    await init_db()
    print("✓ Database tables created/verified")

    # Step 2: Verify schema
    schema_valid = await verify_database_schema()
    if not schema_valid:
        print("⚠️  Warning: Database schema may be incomplete")
        return False

    print("✓ Database schema validated")

    # Step 3: Seed initial data
    await seed_initial_data(force=force_seed)

    return True
