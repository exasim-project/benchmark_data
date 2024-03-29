import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import exasim_plot_helpers as eph

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
    if log == "both":
        plt.xscale("log")
        plt.yscale("log")
    if log == "x":
        plt.xscale("log")
    plt.savefig(post_pro_dir / name)


class Df_filter:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, df):
        return self.func(df)


unprecond = lambda df: df[df["preconditioner"] == "none"]


def compute_speedup(df, bases, extra_filter=lambda df: df):
    from copy import deepcopy

    indices = [q.idx for q in bases[0]["base"]]
    indices.append("Host")
    # things that need to match
    indices += ["nCells"]
    df_copy = deepcopy(extra_filter(df))
    df_copy.set_index(keys=indices, inplace=True)
    return eph.helpers.compute_speedup(df_copy, bases, ignore_indices=[]).reset_index()



bases = [
{
    "case": [
        eph.helpers.DFQuery(idx="Host", val="nla"),
        ],
    "base" : [
        # TODO this needs to know nProcs beforehand
        eph.helpers.DFQuery(idx="nProcs", val=32),
        eph.helpers.DFQuery(idx="preconditioner", val="none"),
        eph.helpers.DFQuery(idx="executor", val="CPU"),
        ]
},
{
    "case": [
        eph.helpers.DFQuery(idx="Host", val="hkn"),
        ],
    "base" : [
        # TODO this needs to know nProcs beforehand
        eph.helpers.DFQuery(idx="nProcs", val=76),
        eph.helpers.DFQuery(idx="preconditioner", val="none"),
        eph.helpers.DFQuery(idx="executor", val="CPU"),
        ]
},
]


for x, c in [("nCells", "nProcs"), ("nProcs", "nCells")]:
    for y in ["TimeStep", "SolveP"]:
        plotter(
            x=x,
            y=y,
            color=c,
            style="solver_p",
            plot_type="line",
            col="Host",
            log=True,
            df=df,
            df_filter=Df_filter("unpreconditioned", unprecond),
        )

        plotter(
            x=x,
            y=y,
            color=c,
            style="solver_p",
            plot_type="line",
            col="Host",
            log=True,
            df=df,
            df_filter=Df_filter(
                "unprecond_speedup", lambda df: compute_speedup(df, bases, unprecond)
            ),
        )
