import os

from foreman.db import base


def test_tests_never_use_the_projects_real_database():
    assert "crowdio-tests-" in base.DATABASE_URL
    assert os.path.abspath("crowdio.db") != base._sqlite_file_path()
    assert str(base.engine.url).endswith("crowdio-test.db")
