import click
import cv2
from rich.console import Console
from rich.table import Table

from ic.stega import harness
from ic.stega import watermark

console = Console()


def _read(path: str):
    """Load a BGR image, erroring clearly if the path isn't a readable image."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise click.ClickException(f"Could not read image: {path}")
    return image


def _signature(signature: str | None) -> bytes:
    """Resolve a signature string to bytes, defaulting to the PB signature."""
    return signature.encode("utf-8") if signature else watermark.PB_SIGNATURE


method_option = click.option(
    "--method", default=watermark.DEFAULT_METHOD,
    type=click.Choice(watermark.METHODS), show_default=True,
)
signature_option = click.option(
    "--signature", default=None,
    help="Signature string to embed/recover (default: the PB signature).",
)
threshold_option = click.option(
    "--threshold", default=watermark.PRESENCE_THRESHOLD, show_default=True,
    help="Bit-accuracy at/above which the watermark counts as present.",
)


@click.group()
def cli():
    """Stega CLI Tool

    Invisible frequency-domain watermarking (DWT-DCT-SVD) proof of concept for
    the Pinkbike BuySell marketplace. Embed an invisible signature into a listing
    image, then detect it later to catch reused/duplicated photos -- even after
    EXIF stripping, resizing, and recompression.
    """
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@signature_option
@method_option
def embed(input, output, signature, method):
    """Embed the ECC-coded PB watermark from INPUT into OUTPUT."""
    sig = _signature(signature)
    result = watermark.embed_watermark(_read(input), sig, method)
    cv2.imwrite(output, result)
    console.print(
        f"[green]Embedded[/green] signature {sig!r} (x{watermark.REPETITIONS} ECC) "
        f"via [cyan]{method}[/cyan] -> [bold]{output}[/bold]"
    )


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@signature_option
@method_option
@threshold_option
def detect(input, signature, method, threshold):
    """Detect the PB watermark in INPUT and report presence + recovery."""
    sig = _signature(signature)
    r = watermark.detect_watermark(_read(input), sig, method, threshold)
    if r.error:
        console.print(f"[red]Decode failed:[/red] {r.error}")
        return
    verdict = "[green]PRESENT[/green]" if r.present else "[red]absent[/red]"
    exact = "[green]exact[/green]" if r.exact_match else "[yellow]ECC near-miss[/yellow]"
    console.print(
        f"{verdict}  bit_accuracy={r.bit_accuracy:.1%} (threshold {threshold:.0%})  "
        f"recovered={r.recovered!r} ({exact})"
    )


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@signature_option
@method_option
@threshold_option
def attack(input, signature, method, threshold):
    """Embed in INPUT, then measure watermark survival against edit attacks."""
    sig = _signature(signature)
    outcomes = harness.run_gauntlet(_read(input), sig, method, threshold)

    table = Table(title=f"Watermark robustness ({method}, signature={sig!r}, "
                        f"threshold={threshold:.0%})")
    table.add_column("Attack")
    table.add_column("Present", justify="center")
    table.add_column("Bit acc", justify="right")
    table.add_column("Control", justify="right")
    table.add_column("Recovered")
    for o in outcomes:
        r = o.result
        present = "[green]yes[/green]" if r.present else "[red]no[/red]"
        recovered = r.error or repr(r.recovered)
        if not r.error and r.exact_match:
            recovered += " [green](exact)[/green]"
        table.add_row(
            o.name, present, f"{r.bit_accuracy:.1%}",
            f"{o.control_accuracy:.1%}", recovered,
        )
    console.print(table)
