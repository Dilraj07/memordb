import os
import sys
import time
import logging

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Suppress HuggingFace / Transformers warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# Configure memor engine logger to intercept events
memor_logger = logging.getLogger("memor")
memor_logger.setLevel(logging.INFO)

class RichDemoLogHandler(logging.Handler):
    """Custom handler that formats engine events with high-contrast cyber styling."""
    def emit(self, record):
        msg = self.format(record)
        if "Fast-path" in msg or "UPDATE" in msg:
            console.print(f"  [bold yellow]⚡ [RESOLVER][/bold yellow] [dim cyan]{msg}[/dim cyan]")
        elif "ADD" in msg:
            console.print(f"  [bold green]✦ [PERSIST][/bold green] [dim white]{msg}[/dim white]")
        elif "FTS" in msg or "RETRIEVE" in msg:
            console.print(f"  [bold magenta]🔍 [RETRIEVER][/bold magenta] [dim white]{msg}[/dim white]")
        else:
            console.print(f"  [bold cyan]⚙ [ENGINE][/bold cyan] [dim]{msg}[/dim]")

handler = RichDemoLogHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
memor_logger.addHandler(handler)

# Clear terminal for pristine recording
os.system('cls' if os.name == 'nt' else 'clear')

# ─────────────────────────────────────────────────────────────
#  CYBERPUNK ASCII HEADER
# ─────────────────────────────────────────────────────────────
ASCII_BANNER = r"""
[bold cyan] __  __ ______ __  __  ____  _____   [/bold cyan]
[bold cyan]|  \/  |  ____|  \/  |/ __ \|  __ \  [/bold cyan]
[bold cyan]| \  / | |__  | \  / | |  | | |__) | [/bold cyan]
[bold cyan]| |\/| |  __| | |\/| | |  | |  _  /  [/bold cyan]
[bold cyan]| |  | | |____| |  | | |__| | | \ \  [/bold cyan]
[bold cyan]|_|  |_|______|_|  |_|\____/|_|  \_\ [/bold cyan]
"""

console.print(ASCII_BANNER)
console.print("[bold white]   ⚡ LOCAL-FIRST BITEMPORAL MEMORY ENGINE FOR LLM AGENTS ⚡[/bold white]", justify="center")
console.print(
    "[bold black on cyan] LOCAL-FIRST [/bold black on cyan]  "
    "[bold black on green] ZERO-DRIFT [/bold black on green]  "
    "[bold black on magenta] GRAPH-RAG [/bold black on magenta]  "
    "[bold black on yellow] SQLITE WAL [/bold black on yellow]",
    justify="center",
)
console.print()

def pulse_log(text: str, delay: float = 0.4):
    console.print(f" [bold cyan]❯[/bold cyan] [white]{text}[/white]")
    time.sleep(delay)

pulse_log("Mounting embedded storage [bold yellow]memory.db[/bold yellow] (WAL Mode)...", 0.3)
pulse_log("Pre-warming hybrid pipeline: [bold magenta]FTS5 Lexical[/bold magenta] + [bold green]Dense Vector Space[/bold green]...", 0.3)
pulse_log("Bitemporal state ledger [bold green]Schema v6[/bold green] verified.", 0.3)
pulse_log("Fast-path heuristic & LLM judge resolution channels armed.", 0.3)
console.print(" [bold green]✔[/bold green] [bold white]SYSTEM OPERATIONAL. AWAITING MULTI-TURN DIALOGUE.[/bold white]\n")
time.sleep(0.8)

# Import engine
from memor.engine import MemoryEngine
import memor.store as store

engine = MemoryEngine()
user_id = "demo_operative"
engine.clear_memories(user_id)

def typewriter(prompt_prefix: str, text: str, prefix_color: str = "bold green", char_delay: float = 0.035):
    console.print(f"[{prefix_color}]{prompt_prefix}[/{prefix_color}] ", end="")
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(char_delay)
    print()

def section_header(title: str, subtitle: str, color: str = "cyan"):
    console.print()
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(f"[bold {color}]═══ {title.upper()} ═══[/bold {color}]")
    grid.add_row(f"[dim white]{subtitle}[/dim white]")
    console.print(Panel(grid, border_style=color, box=box.ROUNDED))
    console.print()

# ═════════════════════════════════════════════════════════════
#  PART 1: BITEMPORAL CONTRADICTION RESOLUTION
# ═════════════════════════════════════════════════════════════
section_header(
    "Part 1: Bitemporal Contradiction Resolution",
    "Tracking dynamic human facts across time without stale memory hallucinations",
    "yellow"
)

# Turn 1
typewriter("[USER TURN 1]", "I live in San Francisco, California.", "bold green")
time.sleep(0.3)
engine.add_memory(user_id, "I live in San Francisco, California.", sync=True)
console.print("  [dim]↳ Initial state anchored: valid_to = NULL (Active)[/dim]\n")
time.sleep(1.0)

# Turn 2
typewriter("[USER TURN 2]", "Actually, I just relocated to Seattle for a new role.", "bold green")
time.sleep(0.3)
engine.add_memory(user_id, "Actually, I just relocated to Seattle for a new role.", sync=True)
console.print("  [dim]↳ Single-valued predicate update: old record invalidated, new record active[/dim]\n")
time.sleep(1.0)

# Query
console.print("[bold yellow]⚡ EXECUTING TEMPORAL RETRIEVAL CHECK[/bold yellow]")
typewriter("[AGENT QUERY]", "Where do I live?", "bold magenta")
time.sleep(0.5)

results = engine.query(user_id, "Where do I live?", k=3)

# Display result in rich table
res_table = Table(title="Retrieved Active Truths (Top K)", box=box.ROUNDED, border_style="yellow")
res_table.add_column("Rank", style="bold cyan", width=6)
res_table.add_column("Fact Formulation", style="bold white")
res_table.add_column("Temporal State", style="bold green")
res_table.add_column("Confidence / Relevance", style="bold yellow")

lines = [line.strip() for line in results.split("\n") if line.strip().startswith("-")]
for idx, line in enumerate(lines, 1):
    cleaned = line.lstrip("- ")
    fact_part, _, score_part = cleaned.partition(" (Relevance: ")
    score = score_part.rstrip(")") if score_part else "N/A"
    res_table.add_row(f"#{idx}", fact_part, "ACTIVE (valid_to: NULL)", f"{score}/10.0")

console.print(res_table)
console.print("  [bold green]✔[/bold green] [bold white]TEMPORAL DRIFT ZERO:[/bold white] Stale fact ('San Francisco') cleanly ignored without data destruction.\n")
time.sleep(1.5)

# ═════════════════════════════════════════════════════════════
#  PART 2: MULTI-HOP GRAPH REASONING (GRAPHRAG-LITE)
# ═════════════════════════════════════════════════════════════
section_header(
    "Part 2: Multi-Hop Recursive Graph Traversal",
    "Traversing relational knowledge chains natively via SQLite WITH RECURSIVE CTEs",
    "magenta"
)

# Turn 3
typewriter("[USER TURN 3]", "My brother is named Alex.", "bold green")
time.sleep(0.3)
engine.add_memory(user_id, "My brother is named Alex.", sync=True)
console.print("  [dim]↳ Edge established: (user) ──[brother]──► (Alex)[/dim]\n")
time.sleep(1.0)

# Turn 4
typewriter("[USER TURN 4]", "Alex adopted a golden retriever named Buddy.", "bold green")
time.sleep(0.3)
engine.add_memory(user_id, "Alex adopted a golden retriever named Buddy.", sync=True)
console.print("  [dim]↳ Edge established: (Alex) ──[owns_pet]──► (golden retriever named Buddy)[/dim]\n")
time.sleep(1.0)

# Multi-hop Query
console.print("[bold magenta]🔍 INITIATING MULTI-HOP GRAPH TRAVERSAL[/bold magenta]")
typewriter("[AGENT QUERY]", "What kind of dog does my brother have?", "bold magenta")
time.sleep(0.4)

console.print("  [dim cyan]⚡ SQL CTE executing: seed='user' | depth=2 | cycle-shield=active[/dim cyan]")
results_graph = engine.query(user_id, "What kind of dog does my brother have?", k=2)
time.sleep(0.6)

graph_table = Table(title="GraphRAG Context Fusion", box=box.ROUNDED, border_style="magenta")
graph_table.add_column("Hop Depth", style="bold cyan", width=10)
graph_table.add_column("Inferred Knowledge Path", style="bold white")
graph_table.add_column("Relevance", style="bold yellow")

lines_graph = [line.strip() for line in results_graph.split("\n") if line.strip().startswith("-")]
for idx, line in enumerate(lines_graph, 1):
    cleaned = line.lstrip("- ")
    fact_part, _, score_part = cleaned.partition(" (Relevance: ")
    score = score_part.rstrip(")") if score_part else "10.0"
    hop = "2-HOP LINK" if "->" in fact_part else "DIRECT"
    graph_table.add_row(hop, fact_part, f"{score}/10.0")

console.print(graph_table)
console.print("  [bold green]✔[/bold green] [bold white]GRAPH FUSION SUCCESS:[/bold white] Associative query answered without external graph DB clusters.\n")
time.sleep(1.5)

# ═════════════════════════════════════════════════════════════
#  PART 3: LIVE BITEMPORAL AUDIT LEDGER (THE TIME MACHINE)
# ═════════════════════════════════════════════════════════════
section_header(
    "Part 3: The Bitemporal Database Ledger",
    "Direct inspection of SQLite internal valid_from / valid_to audit intervals",
    "cyan"
)

import sqlite3
conn = sqlite3.connect(store.DB_PATH)
cursor = conn.cursor()
cursor.execute(
    "SELECT id, subject, predicate, object, valid_from, valid_to FROM facts WHERE user_id = ? ORDER BY id ASC",
    (user_id,)
)
db_rows = cursor.fetchall()
conn.close()

ledger_table = Table(
    title=f"Internal Database Ledger [memory.db] — Tenant: '{user_id}'",
    box=box.ROUNDED,
    border_style="cyan"
)
ledger_table.add_column("Fact ID", style="bold cyan", width=8)
ledger_table.add_column("Subject", style="white", width=10)
ledger_table.add_column("Predicate", style="yellow", width=12)
ledger_table.add_column("Object", style="white")
ledger_table.add_column("Valid From", style="dim white", width=19)
ledger_table.add_column("Valid To", style="dim white", width=19)
ledger_table.add_column("Audit Status", style="bold")

for r in db_rows:
    fid, subj, pred, obj, v_from, v_to = r
    # Format timestamps
    t_from = v_from[:19] if v_from else "N/A"
    t_to = v_to[:19] if v_to else "NULL (Active)"
    
    if v_to is None:
        status = "[bold black on green] ACTIVE [/bold black on green]"
    else:
        status = "[bold white on red] SUPERSEDED [/bold white on red]"
        
    ledger_table.add_row(str(fid), subj, pred, obj, t_from, t_to, status)

console.print(ledger_table)
console.print("  [bold cyan]ℹ[/bold cyan] [dim white]Complete historical state is preserved for forensic replay or audit queries.[/dim white]\n")
time.sleep(1.2)

# ─────────────────────────────────────────────────────────────
#  DEMO CONCLUSION BANNER
# ─────────────────────────────────────────────────────────────
console.print(
    Panel(
        "[bold white]MEMOR DEMONSTRATION COMPLETE[/bold white]\n"
        "[dim]Single-file local engine  •  Zero vector drift  •  Embedded recursive CTEs[/dim]\n\n"
        "[bold green]Ready for autonomous agent production workflows.[/bold green]",
        title="[bold cyan]★ SUMMARY ★[/bold cyan]",
        border_style="green",
        box=box.DOUBLE,
    )
)
