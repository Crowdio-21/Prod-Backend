#!/usr/bin/env python3
"""
Script to view the contents of the CrowdCompute SQLite database
"""

import sys
import os
import asyncio
import sqlite3
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def view_database():
    """View database contents"""
    try:
        # Connect to SQLite database (database is in project root, not tests directory)
        project_root = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(project_root, 'crowdio.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🗄️  CrowdCompute Database Contents")
        print("=" * 50)
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            print("❌ No tables found. Database may not be initialized.")
            return
        
        print(f"📋 Tables found: {[table[0] for table in tables]}")
        print()
        
        # View jobs table
        print("📊 JOBS TABLE:")
        print("-" * 30)
        try:
            cursor.execute("SELECT * FROM jobs")
            jobs = cursor.fetchall()
            if jobs:
                print(f"Total jobs: {len(jobs)}")
                for job in jobs:
                    print(f"  ID: {job[0]}, Status: {job[1]}, Tasks: {job[2]}/{job[3]}, Created: {job[4]}")
            else:
                print("  No jobs found")
        except sqlite3.OperationalError:
            print("  Table 'jobs' does not exist")
        print()
        
        # View tasks table
        print("📋 TASKS TABLE:")
        print("-" * 30)
        try:
            cursor.execute("SELECT * FROM tasks")
            tasks = cursor.fetchall()
            if tasks:
                print(f"Total tasks: {len(tasks)}")
                for task in tasks:
                    print(f"  ID: {task[0]}, Job: {task[1]}, Worker: {task[2]}, Status: {task[3]}")
            else:
                print("  No tasks found")
        except sqlite3.OperationalError:
            print("  Table 'tasks' does not exist")
        print()
        
        # View workers table
        print("👥 WORKERS TABLE:")
        print("-" * 30)
        try:
            cursor.execute("SELECT * FROM workers")
            workers = cursor.fetchall()
            if workers:
                print(f"Total workers: {len(workers)}")
                for worker in workers:
                    print(f"  ID: {worker[0]}, Status: {worker[1]}, Last seen: {worker[2]}")
            else:
                print("  No workers found")
        except sqlite3.OperationalError:
            print("  Table 'workers' does not exist")
        print()
        
        # View worker_failures table
        print("⚠️  WORKER_FAILURES TABLE:")
        print("-" * 30)
        try:
            cursor.execute("SELECT * FROM worker_failures")
            failures = cursor.fetchall()
            if failures:
                print(f"Total failures: {len(failures)}")
                for failure in failures[:10]:  # Show first 10
                    print(f"  Worker: {failure[1]}, Task: {failure[2]}, Error: {failure[3][:50]}...")
                if len(failures) > 10:
                    print(f"  ... and {len(failures) - 10} more failures")
            else:
                print("  No failures found")
        except sqlite3.OperationalError:
            print("  Table 'worker_failures' does not exist")
        print()
        
        # Show table schemas
        print("🏗️  TABLE SCHEMAS:")
        print("-" * 30)
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"\n{table_name.upper()}:")
            for col in columns:
                print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database error: {e}")
        print("💡 Try running 'python init_database.py' first")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    view_database()
