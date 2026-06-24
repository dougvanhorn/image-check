import click
import cv2
from rich.console import Console
from rich.table import Table

from ic.phash import harness
from ic.phash import signals

console = Console()


def _read(path: str):
    """Load a BGR image, erroring clearly if the path isn't a readable image."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise click.ClickException(f"Could not read image: {path}")
    return image


signal_option = click.option(
    "--signal", "signal_name", default=signals.DEFAULT_SIGNAL,
    type=click.Choice(list(signals.SIGNALS)), show_default=True,
    help="Which geometry-tolerant signal to use.",
)


@click.group()
def cli():
    """Phash CLI Tool

    Spike for the geometry-tolerant duplicate signal (PB-658). Where the `stega`
    watermark dies on resize/crop, these content-derived signals -- perceptual
    hash and CLIP embedding -- match by similarity and tolerate those edits.
    """
    pass


@cli.command(name="hash")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@signal_option
def hash_(input, signal_name):
    """Print the descriptor for INPUT (a hash string, or embedding shape)."""
    signal = signals.get_signal(signal_name)
    desc = signal.descriptor(_read(input))
    shown = desc if signal_name == "phash" else f"vector dim={len(desc)}"
    console.print(f"[cyan]{signal_name}[/cyan]  {shown}")


@cli.command()
@click.argument("a", type=click.Path(exists=True, dir_okay=False))
@click.argument("b", type=click.Path(exists=True, dir_okay=False))
@signal_option
def compare(a, b, signal_name):
    """Report similarity between two images A and B."""
    signal = signals.get_signal(signal_name)
    sim = signal.similarity(signal.descriptor(_read(a)), signal.descriptor(_read(b)))
    match = "[green]match[/green]" if sim >= signal.default_threshold else "[red]no match[/red]"
    console.print(
        f"[cyan]{signal_name}[/cyan]  similarity={sim:.1%}  {match} "
        f"(threshold {signal.default_threshold:.0%})"
    )


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@signal_option
@click.option("--threshold", default=None, type=float,
              help="Match threshold (default: the signal's own).")
@click.option("--save-dir", default=None, type=click.Path(file_okay=False),
              help="Write each attacked test image here as PNG for review.")
def attack(input, signal_name, threshold, save_dir):
    """Measure whether INPUT still matches itself after each edit attack."""
    signal = signals.get_signal(signal_name)
    thr = threshold if threshold is not None else signal.default_threshold
    outcomes = harness.run_gauntlet(_read(input), signal, thr, save_dir=save_dir)

    table = Table(title=f"Geometry tolerance ({signal_name}, threshold={thr:.0%})")
    table.add_column("Attack")
    table.add_column("Matches", justify="center")
    table.add_column("Similarity", justify="right")
    if save_dir:
        table.add_column("Saved")
    for o in outcomes:
        matches = "[green]yes[/green]" if o.matched else "[red]no[/red]"
        sim = o.error or f"{o.similarity:.1%}"
        row = [o.name, matches, sim]
        if save_dir:
            row.append(o.saved_path or "")
        table.add_row(*row)
    console.print(table)
    if save_dir:
        console.print(f"[green]Saved {len(outcomes)} test images to[/green] {save_dir}")
