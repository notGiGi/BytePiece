from typing import List, Literal
from rich.console import Console
from rich.text import Text
from rich.table import Table


class ChunkViewer:
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def show_tokens(
        self,
        text: str,
        tokens: List[str],
        style: Literal["inline", "table", "colored"] = "inline"
    ):
        if style == "inline":
            self._show_inline(tokens)
        elif style == "table":
            self._show_table(text, tokens)
        elif style == "colored":
            self._show_colored(tokens)
    
    def _show_inline(self, tokens: List[str]):
        self.console.print("\n[bold cyan]Tokens:[/bold cyan]")
        formatted = " | ".join(f"[green]{t}[/green]" for t in tokens)
        self.console.print(f"  {formatted}")
        self.console.print(f"\n[dim]Total: {len(tokens)} tokens[/dim]\n")
    
    def _show_table(self, text: str, tokens: List[str]):
        table = Table(title="Token Analysis", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Token", style="green")
        table.add_column("Bytes", justify="right", style="cyan")
        table.add_column("Type", style="yellow")
        
        for i, token in enumerate(tokens, 1):
            token_type = self._classify_token(token)
            byte_len = len(token.encode('utf-8'))
            table.add_row(str(i), token, str(byte_len), token_type)
        
        self.console.print(table)
        self.console.print(f"\n[bold]Summary:[/bold] {len(tokens)} tokens, "
                          f"{len(text)} chars → "
                          f"{len(tokens)/len(text):.2%} compression\n")
    
    def _show_colored(self, tokens: List[str]):
        self.console.print("\n[bold cyan]Color-coded tokens:[/bold cyan]")
        
        colors = {
            "keyword": "bold magenta",
            "operator": "bold red",
            "identifier": "green",
            "literal": "yellow",
            "other": "white",
        }
        
        text = Text()
        for token in tokens:
            token_type = self._classify_token(token)
            color = colors.get(token_type, "white")
            text.append(f"[{token}]", style=color)
            text.append(" ")
        
        self.console.print(text)
        self.console.print()
    
    def _classify_token(self, token: str) -> str:
        keywords = {"def", "class", "if", "else", "return", "import", "from", "for"}
        if token.strip("▁ ") in keywords:
            return "keyword"
        
      
        operators = {"=", "==", ">=", "<=", "+", "-", "*", "/", ":", "(", ")"}
        if token.strip() in operators:
            return "operator"
        
       
        if token.startswith('"') or token.startswith("'") or token.isdigit():
            return "literal"
        
      
        if token.replace("▁", "").isidentifier():
            return "identifier"
        
        return "other"


def show_tokens(
    encoder,
    text: str,
    style: Literal["inline", "table", "colored"] = "inline"
):
    tokens = encoder.encode(text)
 
    if tokens and isinstance(tokens[0], int):
        tokens = [encoder.vocab.id_to_token.get(tid, f"<{tid}>") for tid in tokens]
    
    viewer = ChunkViewer()
    viewer.show_tokens(text, tokens, style=style)