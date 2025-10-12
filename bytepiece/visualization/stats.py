from typing import List, Dict
from collections import Counter
from rich.console import Console
from rich.table import Table
import statistics


class TokenStats:
   
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def analyze(self, tokens: List[str]) -> Dict:
        
        if not tokens:
            return {}
        
        byte_lengths = [len(t.encode('utf-8')) for t in tokens]
        char_lengths = [len(t) for t in tokens]
        
        stats = {
            "total_tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "avg_bytes": statistics.mean(byte_lengths),
            "median_bytes": statistics.median(byte_lengths),
            "max_bytes": max(byte_lengths),
            "min_bytes": min(byte_lengths),
            "avg_chars": statistics.mean(char_lengths),
            "most_common": Counter(tokens).most_common(10),
        }
        
        return stats
    
    def show_stats(self, tokens: List[str]):
   
        stats = self.analyze(tokens)
        
        self.console.print("\n[bold cyan] Token Statistics[/bold cyan]\n")
        
        # Tabla de métricas
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="yellow")
        table.add_column("Value", style="green")
        
        table.add_row("Total tokens", str(stats["total_tokens"]))
        table.add_row("Unique tokens", str(stats["unique_tokens"]))
        table.add_row("Avg bytes/token", f"{stats['avg_bytes']:.2f}")
        table.add_row("Median bytes/token", f"{stats['median_bytes']:.1f}")
        table.add_row("Token size range", 
                     f"{stats['min_bytes']}-{stats['max_bytes']} bytes")
        
        self.console.print(table)
        
        
        if stats["most_common"]:
            self.console.print("\n[bold yellow]Most Common Tokens:[/bold yellow]")
            for token, count in stats["most_common"][:5]:
                self.console.print(f"  • {token!r}: {count}x")
        
        self.console.print()