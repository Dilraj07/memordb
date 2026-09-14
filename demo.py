import time
import os
import sys

# Suppress HuggingFace warnings for a clean demo recording
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

# Enable memor logging so we can see real engine output
import logging
memor_logger = logging.getLogger("memor")
memor_logger.setLevel(logging.INFO)

class DemoLogHandler(logging.Handler):
    """Custom handler that formats engine logs as system messages."""
    def emit(self, record):
        msg = self.format(record)
        print(f"\033[1;36m  [Engine]\033[0m {msg}")

handler = DemoLogHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
memor_logger.addHandler(handler)

# Clear screen for demo
os.system('cls' if os.name == 'nt' else 'clear')

print("\033[1;36m[System]\033[0m Initializing Memor Bitemporal Engine...")
time.sleep(1)

from memor.engine import MemoryEngine
engine = MemoryEngine()
user_id = "demo_user"

# Wipe clear for the demo
engine.clear_memories(user_id)

def type_text(text, color="\033[1;37m", prefix="\033[1;32m[User]\033[0m "):
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(f"{color}{char}\033[0m")
        sys.stdout.flush()
        time.sleep(0.04)
    print()

def print_system(text):
    print(f"\033[1;36m[System]\033[0m {text}")
    time.sleep(0.8)

# ═══════════════════════════════════════════════════════
#  PART 1: Bitemporal Contradiction Resolution
# ═══════════════════════════════════════════════════════
print("\n" + "═"*60)
print("\033[1;33m  Part 1: Bitemporal Contradiction Resolution\033[0m")
print("═"*60 + "\n")

type_text("I live in San Francisco, California.")
engine.add_memory(user_id, "I live in San Francisco, California.", sync=True)
print()

print("-"*60 + "\n")

type_text("Actually, I just moved to Seattle for a new job.")
engine.add_memory(user_id, "Actually, I just moved to Seattle for a new job.", sync=True)
print()

print("-"*60 + "\n")
print_system("Retrieving Active State (Zero Temporal Drift)...")
time.sleep(1)

type_text("Where do I live?", prefix="\033[1;35m[Query]\033[0m ")
results = engine.query(user_id, "Where do I live?", k=1)

print(f"\n\033[1;33m[Engine Response]\033[0m\n{results}")

# ═══════════════════════════════════════════════════════
#  PART 2: Multi-Hop Graph Traversal
# ═══════════════════════════════════════════════════════
print("\n\n" + "═"*60)
print("\033[1;33m  Part 2: Multi-Hop Graph Traversal\033[0m")
print("═"*60 + "\n")

type_text("My brother is named Alex.")
engine.add_memory(user_id, "My brother is named Alex.", sync=True)
print()

type_text("Alex adopted a golden retriever named Buddy.")
engine.add_memory(user_id, "Alex adopted a golden retriever named Buddy.", sync=True)
print()

print("-"*60 + "\n")
print_system("Traversing relationship graph via WITH RECURSIVE CTE...")
time.sleep(1)

type_text("What kind of dog does my brother have?", prefix="\033[1;35m[Query]\033[0m ")
results = engine.query(user_id, "What kind of dog does my brother have?", k=2)

print(f"\n\033[1;33m[Engine Response]\033[0m\n{results}")

print("\n" + "═"*60 + "\n")
print("\033[1;32mDemo Complete.\033[0m")
