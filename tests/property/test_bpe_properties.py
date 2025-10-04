"""BytePiece command-line interface."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import bytepiece
from bytepiece.core.normalizer import NormalizationMode, SpacerMode

app = typer.Typer(
    name="bytepiece",
    help="Educational BPE and Unigram tokenizer",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    corpus: Path = typer.Argument(..., help="Path to training corpus"),
    output: Path = typer.Argument(..., help="Output model path"),
    vocab_size: int = typer.Option(1000, "--vocab-size", "-v", help="Target vocabulary size"),
    normalization: NormalizationMode = typer.Option(
        NormalizationMode.NFKC, "--normalization", "-n", help="Unicode normalization mode"
    ),
    spacer_mode: SpacerMode = typer.Option(
        SpacerMode.PREFIX, "--spacer", "-s", help="Spacer mode for word boundaries"
    ),
    byte_fallback: bool = typer.Option(True, "--byte-fallback/--no-byte-fallback", help="Enable byte-fallback"),
    lowercase: bool = typer.Option(False, "--lowercase", help="Convert to lowercase"),
    verbose: bool = typer.Option(False, "--verbose", help="Print training progress"),
):
    """Train a BPE tokenizer on a corpus."""
    console.print(f"[bold blue]Training BPE tokenizer...[/bold blue]")
    console.print(f"Corpus: {corpus}")
    console.print(f"Target vocab size: {vocab_size}")
    
    # Load corpus
    if not corpus.exists():
        console.print(f"[bold red]Error: Corpus file not found: {corpus}[/bold red]")
        raise typer.Exit(1)
    
    with open(corpus, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    console.print(f"Loaded {len(texts)} lines from corpus")
    
    # Create normalizer
    normalizer = bytepiece.Normalizer(
        normalization=normalization,
        spacer_mode=spacer_mode,
        lowercase=lowercase,
    )
    
    # Train model
    vocab, merge_rules, normalizer = bytepiece.train_bpe(
        texts=texts,
        vocab_size=vocab_size,
        normalizer=normalizer,
        byte_fallback=byte_fallback,
        verbose=verbose,
    )
    
    # Create encoder
    encoder = bytepiece.BPEEncoder(
        vocab=vocab,
        merge_rules=merge_rules,
        normalizer=normalizer,
    )
    
    # Save model
    metadata = {
        "corpus_path": str(corpus),
        "num_training_texts": len(texts),
    }
    bytepiece.save_model(encoder, str(output), metadata=metadata)
    
    console.print(f"[bold green]✓[/bold green] Model saved to {output}")
    console.print(f"Final vocab size: {len(vocab)}")
    console.print(f"Num merges: {len(merge_rules)}")


@app.command()
def apply(
    model: Path = typer.Argument(..., help="Path to trained model"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="Input file (stdin if not specified)"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (stdout if not specified)"),
):
    """Apply tokenizer to text."""
    # Load model
    if not model.exists():
        console.print(f"[bold red]Error: Model file not found: {model}[/bold red]")
        raise typer.Exit(1)
    
    encoder = bytepiece.load_model(str(model))
    
    # Read input
    if input_file and input_file.exists():
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        import sys
        texts = [line.strip() for line in sys.stdin if line.strip()]
    
    # Tokenize
    tokenized = encoder.encode_batch(texts)
    
    # Write output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for tokens in tokenized:
                f.write(' '.join(tokens) + '\n')
        console.print(f"[bold green]✓[/bold green] Tokenized {len(texts)} texts to {output_file}")
    else:
        for tokens in tokenized:
            print(' '.join(tokens))


@app.command()
def inspect(
    model: Path = typer.Argument(..., help="Path to trained model"),
    top_merges: int = typer.Option(20, "--top-merges", help="Number of top merges to show"),
):
    """Inspect model configuration and vocabulary."""
    if not model.exists():
        console.print(f"[bold red]Error: Model file not found: {model}[/bold red]")
        raise typer.Exit(1)
    
    # Get model info
    info = bytepiece.get_model_info(str(model))
    
    # Display basic info
    console.print("\n[bold]Model Information[/bold]")
    console.print(f"Algorithm: {info['algorithm']}")
    console.print(f"Version: {info['version']}")
    console.print(f"Created: {info['created_at']}")
    console.print(f"Vocab size: {info['vocab_size']}")
    console.print(f"Num merges: {info['num_merges']}")
    console.print(f"Byte fallback: {info['byte_fallback']}")
    console.print(f"Model hash: {info['model_hash'][:16]}...")
    
    # Display normalizer config
    console.print("\n[bold]Normalizer Configuration[/bold]")
    norm = info['normalizer']
    console.print(f"Normalization: {norm['normalization']}")
    console.print(f"Spacer mode: {norm['spacer_mode']}")
    console.print(f"Lowercase: {norm.get('lowercase', False)}")
    
    # Display top merges
    encoder = bytepiece.load_model(str(model))
    console.print(f"\n[bold]Top {top_merges} Merges[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Left", width=20)
    table.add_column("Right", width=20)
    table.add_column("Result", style="green")
    
    for i, (left, right) in enumerate(encoder.merge_rules.merges[:top_merges]):
        table.add_row(str(i), left, right, left + right)
    
    console.print(table)


@app.command()
def explain(
    text: str = typer.Argument(..., help="Text to tokenize"),
    model: Path = typer.Argument(..., help="Path to trained model"),
):
    """Show step-by-step tokenization process."""
    if not model.exists():
        console.print(f"[bold red]Error: Model file not found: {model}[/bold red]")
        raise typer.Exit(1)
    
    encoder = bytepiece.load_model(str(model))
    
    console.print(f"\n[bold]Input text:[/bold] {text}")
    
    # Normalize
    normalized = encoder.normalizer.normalize(text)
    console.print(f"[bold]After normalization:[/bold] {normalized}")
    
    # Initial tokenization
    tokens = encoder.vocab.encode_with_fallback(normalized)
    console.print(f"\n[bold]Initial tokens (character-level):[/bold]")
    console.print(' '.join(tokens))
    
    # Apply merges step by step
    console.print(f"\n[bold]Applying merges:[/bold]")
    step = 0
    while True:
        best_merge = None
        best_rank = float('inf')
        best_pos = -1
        
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = encoder.merge_rules.get_rank(pair)
            
            if rank is not None and rank < best_rank:
                best_merge = pair
                best_rank = rank
                best_pos = i
        
        if best_merge is None:
            break
        
        step += 1
        console.print(f"  Step {step}: Merge '{best_merge[0]}' + '{best_merge[1]}' → '{best_merge[0] + best_merge[1]}' (rank {best_rank})")
        
        tokens = (
            tokens[:best_pos] +
            [tokens[best_pos] + tokens[best_pos + 1]] +
            tokens[best_pos + 2:]
        )
    
    console.print(f"\n[bold]Final tokens:[/bold]")
    console.print(' '.join(tokens))
    console.print(f"\n[bold]Total tokens:[/bold] {len(tokens)}")


@app.command()
def version():
    """Show version information."""
    console.print(f"BytePiece version {bytepiece.__version__}")


def main():
    """Main entrypoint."""
    app()


if __name__ == "__main__":
    main()