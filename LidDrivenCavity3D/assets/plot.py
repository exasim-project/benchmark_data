import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from pathlib import Path

script_dir = Path(__file__).parent
post_pro_dir = script_dir / "../postProcessing/ogl_170"
json_file = post_pro_dir / "results.json"
df = pd.read_json(json_file)


def plotter(
    x, y, color, style, df, df_filter, size=None, col=None, log=None, plot_type="line"
):
    df = df_filter(df)

    relplot = sb.relplot(
        x=x,
        y=y,
        hue=color,
        size=size,
        style=style,
        data=df,
        col=col,
        kind=plot_type,
        markers=True,
    )
    name = f"{df_filter.name}_{y}_over_{x}_c={color}_s={style}_cols={col}.png"
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(post_pro_dir / name)


class Df_filter:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, df):
        return self.func(df)


plotter(
    x="nCells",
    y="TimeStep",
    color="nProcs",
    style="solver_p",
    plot_type="line",
    col="Host",
    log=True,
    df=df,
    df_filter=Df_filter(
        "unpreconditioned", lambda df: df[df["preconditioner"] == "none"]
    ),
)
