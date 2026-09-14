import pytest
import os
import memor.store as store
from memor.testing.memor_adapter import MemorEngineAdapter


def pytest_configure():
    test_db_path = "benchmark_test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    store.DB_PATH = test_db_path


@pytest.fixture(scope="session")
def memor_adapter():
    adapter = MemorEngineAdapter()
    yield adapter
    adapter.shutdown()
    
    test_db_path = "benchmark_test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
