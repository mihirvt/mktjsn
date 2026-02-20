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
            # Step 1: Safely check if alembic_version table exists without triggering transaction aborts
            res = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_name='alembic_version'"))
            if not res.scalar():
                print("No alembic_version table found. Fresh install. Skipping fix.")
                return

            result = await conn.execute(text("DELETE FROM alembic_version WHERE version_num = 'fefdd1835c9e'"))
            if result.rowcount > 0:
                print(f"Deleted {result.rowcount} ghost row(s) from alembic_version.")
            
            # Step 2: Check if users table exists to gauge DB progress
            res_users = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_name='users'"))
            if not res_users.scalar():
                print("No users table found. DB is fresh. Skipping tracking updates.")
                return

            # Step 3: Check if the upstream migration (6fd8fac02883) was ACTUALLY executed
            has_pwd = await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='password_hash'"))
            is_up_to_date = bool(has_pwd.scalar())

            if not is_up_to_date:
                print("DB is missing password_hash! Ensuring Alembic tracking is set to previous node (6d2f94baf4b7) to trigger creation.")
                await conn.execute(text("UPDATE alembic_version SET version_num = '6d2f94baf4b7' WHERE version_num = '6fd8fac02883'"))
                
                count_res = await conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
                if count_res.scalar() == 0:
                    await conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('6d2f94baf4b7')"))
            else:
                count_res = await conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
                if count_res.scalar() == 0:
                    await conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('6fd8fac02883')"))
                    print("Inserted true upstream HEAD '6fd8fac02883' into empty alembic_version table.")
                else:
                    print(f"alembic_version has {count_res.scalar()} row(s) and DB is fully upgraded. No injection needed.")
                
    except Exception as e:
        print(f"Error fixing database version: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(fix())
