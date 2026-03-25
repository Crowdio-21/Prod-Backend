#!/usr/bin/env python3
"""
Utility script to view and retrieve checkpoints from the database
"""

import os
import sys
import json
import gzip
import pickle
import sqlite3
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_db_connection():
    """Get connection to the crowdio database"""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'crowdio.db')
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)


def list_tasks_with_checkpoints():
    """List all tasks that have checkpoint data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id,
            job_id,
            status,
            progress_percent,
            checkpoint_count,
            base_checkpoint_size,
            checkpoint_storage_path,
            last_checkpoint_at
        FROM tasks 
        WHERE checkpoint_count > 0 OR base_checkpoint_data IS NOT NULL
        ORDER BY last_checkpoint_at DESC
        LIMIT 20
    """)
    
    tasks = cursor.fetchall()
    conn.close()
    
    if not tasks:
        print("📭 No tasks with checkpoints found")
        return []
    
    print("\n" + "=" * 100)
    print("📋 TASKS WITH CHECKPOINTS")
    print("=" * 100)
    
    for task in tasks:
        task_id, job_id, status, progress, cp_count, base_size, storage_path, last_cp = task
        print(f"\n🔹 Task ID: {task_id}")
        print(f"   Job ID: {job_id}")
        print(f"   Status: {status}")
        print(f"   Progress: {progress or 0}%")
        print(f"   Checkpoint Count: {cp_count or 0}")
        print(f"   Base Checkpoint Size: {base_size or 0} bytes")
        print(f"   Storage Path: {storage_path or 'N/A'}")
        print(f"   Last Checkpoint: {last_cp or 'N/A'}")
    
    return tasks


def read_checkpoint_from_database(task_id: str):
    """
    Read checkpoint data from database BLOB columns
    
    Args:
        task_id: The task ID
        
    Returns:
        Dictionary with base state and delta states
    """
    import base64
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            base_checkpoint_data,
            base_checkpoint_blob,
            delta_checkpoints,
            delta_checkpoint_blobs,
            checkpoint_storage_path
        FROM tasks 
        WHERE id = ?
    """, (task_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print(f"❌ Task not found: {task_id}")
        return None
    
    base_ref, base_blob, delta_json, delta_blobs_json, storage_path = row
    
    print("\n" + "=" * 100)
    print(f"📦 READING CHECKPOINT DATA FROM DATABASE FOR TASK: {task_id}")
    print(f"   Storage Path: {storage_path or 'N/A'}")
    print("=" * 100)
    
    result = {
        "base": None,
        "deltas": []
    }
    
    # Read base checkpoint from blob
    if base_blob:
        try:
            decompressed = gzip.decompress(base_blob)
            base_state = pickle.loads(decompressed)
            result["base"] = base_state
            print(f"\n🟢 BASE Checkpoint (from database blob):")
            print(f"   Reference: {base_ref}")
            print(f"   Compressed size: {len(base_blob):,} bytes")
            print(f"   State: {json.dumps(base_state, indent=6, default=str)}")
        except Exception as e:
            print(f"❌ Error reading base blob: {e}")
    else:
        print(f"\n⚠️ No base checkpoint blob in database (may be stored in filesystem)")
        print(f"   Reference: {base_ref or 'N/A'}")
    
    # Read delta checkpoints from blobs
    deltas = json.loads(delta_json or "[]")
    delta_blobs = json.loads(delta_blobs_json or "{}")
    
    print(f"\n🔵 DELTA Checkpoints ({len(deltas)} total):")
    
    for delta in deltas:
        checkpoint_id = str(delta.get("checkpoint_id", ""))
        storage_type = delta.get("storage_type", "unknown")
        
        if checkpoint_id in delta_blobs:
            try:
                blob_b64 = delta_blobs[checkpoint_id]
                blob_data = base64.b64decode(blob_b64)
                decompressed = gzip.decompress(blob_data)
                delta_state = pickle.loads(decompressed)
                result["deltas"].append({
                    "checkpoint_id": checkpoint_id,
                    "state": delta_state
                })
                print(f"\n   [{checkpoint_id}] From database blob ({len(blob_data):,} bytes compressed)")
                print(f"       State: {json.dumps(delta_state, indent=10, default=str)}")
            except Exception as e:
                print(f"\n   [{checkpoint_id}] Error reading delta blob: {e}")
        else:
            print(f"\n   [{checkpoint_id}] Stored in filesystem ({storage_type})")
            print(f"       Reference: {delta.get('storage_ref', 'N/A')}")
    
    return result


def get_task_checkpoint_metadata(task_id: str):
    """Get checkpoint metadata for a specific task"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            base_checkpoint_data,
            base_checkpoint_size,
            delta_checkpoints,
            checkpoint_count,
            checkpoint_storage_path,
            last_checkpoint_at,
            progress_percent
        FROM tasks 
        WHERE id = ?
    """, (task_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print(f"❌ Task not found: {task_id}")
        return None
    
    base_data, base_size, delta_json, cp_count, storage_path, last_cp, progress = row
    
    # Parse delta checkpoints JSON
    deltas = []
    if delta_json:
        try:
            deltas = json.loads(delta_json)
        except json.JSONDecodeError:
            deltas = []
    
    print("\n" + "=" * 100)
    print(f"📊 CHECKPOINT METADATA FOR TASK: {task_id}")
    print("=" * 100)
    print(f"   Progress: {progress or 0}%")
    print(f"   Checkpoint Count: {cp_count or 0}")
    print(f"   Storage Path: {storage_path or 'N/A'}")
    print(f"   Last Checkpoint: {last_cp or 'N/A'}")
    print(f"\n   📦 BASE Checkpoint:")
    print(f"      Reference: {base_data or 'N/A'}")
    print(f"      Size: {base_size or 0} bytes")
    print(f"\n   📦 DELTA Checkpoints ({len(deltas)}):")
    
    for i, delta in enumerate(deltas):
        print(f"      [{i+1}] ID: {delta.get('checkpoint_id', 'N/A')}")
        print(f"          Size: {delta.get('size', 0)} bytes")
        print(f"          Stored At: {delta.get('stored_at', 'N/A')}")
        print(f"          Storage Ref: {delta.get('storage_ref', 'N/A')}")
    
    return {
        "base_checkpoint_data": base_data,
        "base_checkpoint_size": base_size,
        "delta_checkpoints": deltas,
        "checkpoint_count": cp_count,
        "storage_path": storage_path,
        "last_checkpoint_at": last_cp,
        "progress_percent": progress
    }


def retrieve_checkpoint_from_filesystem(task_id: str, checkpoint_dir: str = ".checkpoints"):
    """
    Retrieve checkpoint data from the filesystem storage
    
    Args:
        task_id: The task ID
        checkpoint_dir: Base directory for checkpoint storage
    
    Returns:
        Dictionary with base and delta checkpoint data
    """
    checkpoint_path = os.path.join(checkpoint_dir, task_id)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        return None
    
    print("\n" + "=" * 100)
    print(f"📂 FILESYSTEM CHECKPOINTS FOR TASK: {task_id}")
    print(f"   Directory: {checkpoint_path}")
    print("=" * 100)
    
    result = {
        "base": None,
        "deltas": []
    }
    
    # List all checkpoint files
    files = sorted(os.listdir(checkpoint_path))
    print(f"\n   Files found: {len(files)}")
    
    for filename in files:
        filepath = os.path.join(checkpoint_path, filename)
        file_size = os.path.getsize(filepath)
        
        print(f"\n   📄 {filename} ({file_size:,} bytes)")
        
        try:
            with open(filepath, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress gzip
            decompressed = gzip.decompress(compressed_data)
            
            # Unpickle the state
            state = pickle.loads(decompressed)
            
            print(f"      ✅ Successfully decompressed and unpickled")
            print(f"      State keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")
            
            if isinstance(state, dict):
                for key, value in state.items():
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."
                    print(f"         {key}: {value_str}")
            
            if filename.startswith("base"):
                result["base"] = state
            else:
                result["deltas"].append({"filename": filename, "state": state})
                
        except Exception as e:
            print(f"      ❌ Error reading: {e}")
    
    return result


def list_all_filesystem_checkpoints(checkpoint_dir: str = ".checkpoints"):
    """List all checkpoints stored in the filesystem"""
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    print("\n" + "=" * 100)
    print(f"📂 ALL FILESYSTEM CHECKPOINTS")
    print(f"   Directory: {os.path.abspath(checkpoint_dir)}")
    print("=" * 100)
    
    task_dirs = []
    
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            total_size = sum(os.path.getsize(os.path.join(item_path, f)) for f in files)
            
            print(f"\n   🔹 Task: {item}")
            print(f"      Files: {len(files)}")
            print(f"      Total Size: {total_size:,} bytes")
            print(f"      Contents: {', '.join(files[:5])}" + ("..." if len(files) > 5 else ""))
            
            task_dirs.append({
                "task_id": item,
                "file_count": len(files),
                "total_size": total_size,
                "files": files
            })
    
    if not task_dirs:
        print("\n   📭 No checkpoint directories found")
    
    return task_dirs


def reconstruct_state_from_filesystem(task_id: str, checkpoint_dir: str = ".checkpoints"):
    """
    Reconstruct full state from base + all deltas stored in filesystem
    
    Args:
        task_id: The task ID
        checkpoint_dir: Base directory for checkpoint storage
    
    Returns:
        Reconstructed state dictionary
    """
    checkpoint_path = os.path.join(checkpoint_dir, task_id)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        return None
    
    print("\n" + "=" * 100)
    print(f"🔄 RECONSTRUCTING STATE FOR TASK: {task_id}")
    print("=" * 100)
    
    # Load base checkpoint
    base_file = os.path.join(checkpoint_path, "base.gz")
    if not os.path.exists(base_file):
        print(f"❌ Base checkpoint not found: {base_file}")
        return None
    
    try:
        with open(base_file, 'rb') as f:
            compressed = f.read()
        decompressed = gzip.decompress(compressed)
        base_state = pickle.loads(decompressed)
        print(f"\n🟢 Loaded BASE checkpoint ({len(compressed):,} bytes compressed)")
        print(f"   State: {json.dumps(base_state, indent=6, default=str)}")
    except Exception as e:
        print(f"❌ Error loading base checkpoint: {e}")
        return None
    
    # Find and apply all delta checkpoints
    current_state = dict(base_state) if isinstance(base_state, dict) else base_state
    
    delta_files = sorted([f for f in os.listdir(checkpoint_path) if f.startswith("delta_")])
    
    for delta_file in delta_files:
        delta_path = os.path.join(checkpoint_path, delta_file)
        try:
            with open(delta_path, 'rb') as f:
                compressed = f.read()
            decompressed = gzip.decompress(compressed)
            delta_state = pickle.loads(decompressed)
            
            # Apply delta (merge into current state)
            if isinstance(delta_state, dict) and isinstance(current_state, dict):
                current_state.update(delta_state)
            
            print(f"\n🔵 Applied {delta_file} ({len(compressed):,} bytes)")
            print(f"   Delta: {json.dumps(delta_state, indent=6, default=str)}")
            
        except Exception as e:
            print(f"⚠️ Error applying {delta_file}: {e}")
    
    print(f"\n" + "=" * 100)
    print(f"✅ RECONSTRUCTED STATE:")
    print("=" * 100)
    print(json.dumps(current_state, indent=2, default=str))
    
    return current_state


def show_database_schema():
    """Show the database schema for checkpoint-related tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("\n" + "=" * 100)
    print("🗄️  DATABASE SCHEMA (Checkpoint-Related Fields)")
    print("=" * 100)
    
    # Get tasks table schema
    cursor.execute("PRAGMA table_info(tasks)")
    columns = cursor.fetchall()
    
    print("\n📋 TASKS TABLE - Checkpoint Fields:")
    checkpoint_fields = [
        'base_checkpoint_data', 'base_checkpoint_size', 'base_checkpoint_blob',
        'delta_checkpoints', 'delta_checkpoint_blobs',
        'last_checkpoint_at', 'progress_percent', 'checkpoint_count', 'checkpoint_storage_path'
    ]
    
    for col in columns:
        col_id, name, dtype, notnull, default, pk = col
        if name in checkpoint_fields:
            print(f"   • {name}: {dtype} {'(NOT NULL)' if notnull else '(nullable)'} {'[PK]' if pk else ''}")
            if default is not None:
                print(f"     Default: {default}")
    
    conn.close()


# =========================================================
# 🖥️ COMMAND LINE INTERFACE
# =========================================================

def main():
    """Main function with CLI interface"""
    print("\n" + "=" * 100)
    print("🗄️  CHECKPOINT VIEWER UTILITY")
    print("=" * 100)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python view_checkpoints.py list                      - List all tasks with checkpoints (from DB)")
        print("  python view_checkpoints.py info <task_id>            - Show checkpoint metadata for task")
        print("  python view_checkpoints.py db <task_id>              - Read checkpoint data from database blobs")
        print("  python view_checkpoints.py files                     - List all filesystem checkpoints")
        print("  python view_checkpoints.py read <task_id>            - Read checkpoints from filesystem")
        print("  python view_checkpoints.py reconstruct <task_id>     - Reconstruct state from filesystem")
        print("  python view_checkpoints.py schema                    - Show database schema")
        print("\nExamples:")
        print("  python view_checkpoints.py list")
        print("  python view_checkpoints.py info abc123_task_0")
        print("  python view_checkpoints.py db abc123_task_0          - Read small checkpoints from DB blob")
        print("  python view_checkpoints.py read abc123_task_0")
        print("  python view_checkpoints.py reconstruct abc123_task_0")
        print("\nRunning default: list all tasks with checkpoints\n")
        list_tasks_with_checkpoints()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_tasks_with_checkpoints()
    
    elif command == "info":
        if len(sys.argv) < 3:
            print("❌ Please provide a task ID")
            print("   Usage: python view_checkpoints.py info <task_id>")
            return
        get_task_checkpoint_metadata(sys.argv[2])
    
    elif command == "db":
        if len(sys.argv) < 3:
            print("❌ Please provide a task ID")
            print("   Usage: python view_checkpoints.py db <task_id>")
            return
        read_checkpoint_from_database(sys.argv[2])
    
    elif command == "files":
        list_all_filesystem_checkpoints()
    
    elif command == "read":
        if len(sys.argv) < 3:
            print("❌ Please provide a task ID")
            print("   Usage: python view_checkpoints.py read <task_id>")
            return
        retrieve_checkpoint_from_filesystem(sys.argv[2])
    
    elif command == "reconstruct":
        if len(sys.argv) < 3:
            print("❌ Please provide a task ID")
            print("   Usage: python view_checkpoints.py reconstruct <task_id>")
            return
        reconstruct_state_from_filesystem(sys.argv[2])
    
    elif command == "schema":
        show_database_schema()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("   Run without arguments to see usage help")


if __name__ == "__main__":
    main()
