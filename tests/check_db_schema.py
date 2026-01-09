"""
Quick script to check database schema and worker data
"""

import sqlite3
import os

DATABASE_PATH = "crowdcompute.db"

if not os.path.exists(DATABASE_PATH):
    print(f"❌ Database not found: {DATABASE_PATH}")
    print("💡 Start foreman to create database")
    exit(1)

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Get table schema
cursor.execute("PRAGMA table_info(workers)")
columns = cursor.fetchall()

print("📋 Workers Table Schema:")
print("-" * 80)
for col in columns:
    col_id, name, col_type, not_null, default, pk = col
    pk_marker = " (PRIMARY KEY)" if pk else ""
    print(f"{col_id:2d}. {name:25s} {col_type:15s}{pk_marker}")
print("-" * 80)
print(f"Total columns: {len(columns)}\n")

# Check if device spec columns exist
device_columns = ['device_type', 'os_type', 'cpu_model', 'ram_total_mb', 'battery_level']
existing = [col[1] for col in columns]
missing = [col for col in device_columns if col not in existing]

if missing:
    print(f"⚠️ Missing device spec columns: {', '.join(missing)}")
    print("💡 Delete crowdcompute.db and restart foreman to recreate with new schema")
else:
    print("✅ All device spec columns present!")

# Get worker data
cursor.execute("SELECT * FROM workers")
workers = cursor.fetchall()

if workers:
    print(f"\n👷 Workers in database: {len(workers)}")
    print("-" * 80)
    col_names = [col[1] for col in columns]
    
    for worker in workers:
        print(f"\nWorker: {worker[0]}")
        for i, value in enumerate(worker[1:], 1):
            if value is not None:
                print(f"  {col_names[i]:25s}: {value}")
        print("-" * 80)
else:
    print("\n📭 No workers in database yet")

conn.close()
