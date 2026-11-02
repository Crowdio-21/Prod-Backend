"""Test configuration.

Every test run uses a throwaway database. Without this, tests that touch the database wrote
into the project's real ``crowdio.db`` whenever an earlier test left a pooled connection open
against it (the tests ``chdir`` to temp folders, but a pooled SQLite connection keeps the file
it first opened). This must run before any ``foreman`` module is imported, so it sets the
variable at import time and always overrides an existing value.
"""

import atexit
import os
import shutil
import tempfile

_DB_DIR = tempfile.mkdtemp(prefix="crowdio-tests-")
os.environ["CROWDIO_DATABASE_URL"] = "sqlite+aiosqlite:///" + os.path.join(_DB_DIR, "crowdio-test.db").replace("\\", "/")
atexit.register(shutil.rmtree, _DB_DIR, ignore_errors=True)
