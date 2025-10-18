import typer
from pathlib import Path
from marbl.core.config import load_config
from marbl.io.csv import load_timeseries_csv
from marbl.processing.quality import coverage_pct
from marbl.features.spikes import detect_spikes

app = typer.Typer(help="MARBL energy CLI")


@app.command()
def show_config(path: str = "config/project.yml"):
    cfg = load_config(path)
    typer.echo(cfg.model_dump_json(indent=2, by_alias=True))


@app.command()
def quality(csv: str, col: str):
    df = load_timeseries_csv(Path(csv))
    pct = coverage_pct(df, col)
    typer.echo(f"coverage_pct({col}) = {pct:.2f}%")


@app.command()
def spikes(csv: str, col: str, pos: float = 250.0, neg: float = -50.0, q: float = 0.99):
    df = load_timeseries_csv(Path(csv))
    res = detect_spikes(df[col], abs_pos=pos, abs_neg=neg, quantile=q)
    typer.echo(f"spikes total = {int(res['is_spike'].sum())}")


if __name__ == "__main__":
    app()
