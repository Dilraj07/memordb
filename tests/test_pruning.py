import pytest
import os
import sqlite3
from memor.engine import MemoryEngine
import memor.store as store

def setup_module(module):
    test_db_path = "prune_test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    store.DB_PATH = test_db_path

def teardown_module(module):
    test_db_path = "prune_test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_prune_history(monkeypatch):
    from memor.extractor import Fact

    def mock_extract_facts(text):
        if "Boston" in text:
            return [Fact(subject="user", predicate="lives_in", object="Boston")]
        return [Fact(subject="user", predicate="lives_in", object="Seattle")]

    monkeypatch.setattr("memor.engine.extract_facts", mock_extract_facts)

    engine = MemoryEngine()
    user_id = "prune_user"
    
    # Add an initial fact
    engine.add_memory(user_id, "I live in Boston.", sync=True)
    
    # Overwrite it to create an inactive historical row
    engine.add_memory(user_id, "I just moved to Seattle.", sync=True)
    
    # Manually check the database to confirm we have 2 rows in 'facts' (1 active, 1 inactive)
    with store._conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM facts WHERE user_id=?", (user_id,)).fetchone()[0]
        assert count == 2, "Should have 2 rows (1 active, 1 inactive) before prune."
        
        # Verify facts_archive is empty
        archive_count = conn.execute("SELECT COUNT(*) FROM facts_archive").fetchone()[0]
        assert archive_count == 0, "Archive should be empty."

    # Since the rows were just created, their valid_to is NOW. 
    # Calling prune(older_than_days=1) will delete nothing.
    archived = engine.prune(older_than_days=1)
    assert archived == 0, "Should not prune facts less than 1 day old."
    
    # Calling prune(older_than_days=0) will delete the inactive row immediately.
    archived = engine.prune(older_than_days=0)
    assert archived == 1, "Should archive exactly 1 inactive fact."
    
    # Verify the hot table was pruned
    with store._conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM facts WHERE user_id=?", (user_id,)).fetchone()[0]
        assert count == 1, "Should have 1 active row left in facts table."
        
        # Verify the archive table received the row
        archive_count = conn.execute("SELECT COUNT(*) FROM facts_archive").fetchone()[0]
        assert archive_count == 1, "Archive should contain 1 row."
        
    engine.shutdown()
