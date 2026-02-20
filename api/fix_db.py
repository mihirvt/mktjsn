import asyncio
import os
import sys

async def fix():
    url = os.environ.get('DATABASE_URL')
    if not url:
        print("DATABASE_URL not found. Skipping DB fix.")
        return

    # Use asyncpg directly or sqlalchemy
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
    except ImportError:
        print("SQLAlchemy not found, skipping DB fix.")
        return

    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            # Delete the ghost revision causing the 'Can't locate revision' error
            # We use DELETE because the true upstream revision may already exist, which would cause an UPDATE to throw a UniqueViolation error.
            result = await conn.execute(text("DELETE FROM alembic_version WHERE version_num = 'fefdd1835c9e'"))
            print(f"Deleted {result.rowcount} ghost row(s) from alembic_version to fix migration chain.")
    except Exception as e:
        print(f"Error fixing database version: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(fix())
