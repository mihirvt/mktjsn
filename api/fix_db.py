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
            # Step 1: Delete the ghost revision if it exists
            result = await conn.execute(text("DELETE FROM alembic_version WHERE version_num = 'fefdd1835c9e'"))
            if result.rowcount > 0:
                print(f"Deleted {result.rowcount} ghost row(s) from alembic_version.")
            
            # Step 2: Ensure the true upstream HEAD is present
            # If the table is empty (because we deleted the only head), insert the true one.
            count_result = await conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
            count = count_result.scalar()
            
            if count == 0:
                await conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('6fd8fac02883')"))
                print("Inserted true upstream HEAD '6fd8fac02883' into empty alembic_version table.")
            else:
                print(f"alembic_version has {count} row(s). No injection needed.")
                
    except Exception as e:
        print(f"Error fixing database version: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(fix())
