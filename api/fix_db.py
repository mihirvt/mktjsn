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
            # We explicitly update the alembic_version table to prevent "Can't locate revision identified by 'fefdd1835c9e'"
            result = await conn.execute(text("UPDATE alembic_version SET version_num = '6fd8fac02883' WHERE version_num = 'fefdd1835c9e'"))
            print(f"Altered {result.rowcount} row(s) in alembic_version to match the true upstream HEAD.")
    except Exception as e:
        print(f"Error fixing database version: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(fix())
