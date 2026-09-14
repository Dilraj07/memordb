import sys
import logging
from memor.engine import MemoryEngine
from memor.store import init_db

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: The 'mcp' library is required to run the memor-server.", file=sys.stderr)
    print("Please install it with: pip install memor-db[mcp]", file=sys.stderr)
    sys.exit(1)

# Disable standard logging to stdout because MCP stdio uses stdout for JSON-RPC communication
logging.getLogger("memor").setLevel(logging.CRITICAL)

# Create the MCP server
mcp = FastMCP("memor-db")

# Initialize the global engine instance (singleton-like for the lifecycle of the server)
engine = None

@mcp.tool()
def add_memory(user_id: str, text: str) -> str:
    """
    Extract factual statements from the user's text, resolve any contradictions 
    with their existing memories, and persist the new state bitemporally.
    """
    global engine
    if engine is None:
        engine = MemoryEngine()
        
    engine.add_memory(user_id, text, sync=True)
    return f"Successfully processed and stored memory for user {user_id}."

@mcp.tool()
def query_memory(user_id: str, text: str, limit: int = 5) -> str:
    """
    Perform a hybrid search across the user's memories (vector similarity + BM25 + entity overlap)
    and recursively traverse graph relations to answer the given text query.
    """
    global engine
    if engine is None:
        engine = MemoryEngine()
        
    results = engine.query(user_id, text, k=limit)
    
    if not results:
        return "No relevant facts found in memory."
        
    return "\n".join(f"- {r}" for r in results)

@mcp.tool()
def get_user_profile(user_id: str) -> str:
    """
    Retrieve all currently active facts and memories for a user.
    Useful for getting a complete summary of the user's profile before starting a conversation.
    """
    global engine
    if engine is None:
        engine = MemoryEngine()
        
    memories = engine.get_all_memories(user_id)
    if not memories:
        return f"No active memories found for user {user_id}."
    
    return "\n".join(f"- {m}" for m in memories)

@mcp.tool()
def forget_memory(user_id: str) -> str:
    """
    Hard-delete ALL memories for a specific user.
    Use this for explicit privacy requests (e.g. 'forget everything you know about me').
    """
    global engine
    if engine is None:
        engine = MemoryEngine()
        
    engine.clear_memories(user_id)
    return f"All memories for user {user_id} have been permanently deleted."

def main():
    """Entry point for the memor-server command line tool."""
    # Ensure the DB schema is initialized
    init_db()
    # Run the stdio server
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
