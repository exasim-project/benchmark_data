import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import exasim_plot_helpers as eph
from pathlib import Path
from copy import deepcopy

from pathlib import Path

def plot_script():
    return """import pandas as pd

df_json = '{}'
df = pd.DataFrame.read_json(df_json)
    """

def plotter(
    x, y, color, style, df, df_filter, post_pro_dir, postfix="", size=None, col=None, log=None, plot_type="line"
):
    df = df_filter(df)
    name = f"{df_filter.name}_{y}_over_{x}_c={color}_s={style}_cols={col}{postfix}"
    script_name = name+".py"

    with open (post_pro_dir / script_name, "w") as script:
        script.write(plot_script().format(df.to_json()))

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
    if log == "both":
        plt.xscale("log")
        plt.yscale("log")
    if log == "x":
        plt.xscale("log")
    fig_name = name + ".png"
    plt.savefig(post_pro_dir / fig_name )



def col_divide(df_orig, df_comparisson):
    ret = deepcopy(df_orig).set_index("jobid")
    df_orig = df_orig.set_index("jobid")
    df_comparisson = df_comparisson.set_index("jobid")
    for c in df_orig.columns:
        if c == "nCells" or c == "nProcs":
            continue
        try:
            print(f"c: {c} ret[c] {df_comparisson[c] / df_orig[c]}  comp:  {df_comparisson[c]} orig: {df_orig[c]}")  
            ret[c] = df_comparisson[c] / df_orig[c]  
        except Exception as e:
            print(e, c)
            pass
    print(f"df[TimeStep] {ret['TimeStep']}")
    return ret



class Df_filter:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, df):
        return self.func(df)


def compute_speedup(df, bases, extra_filter=lambda df: df):
    from copy import deepcopy

    indices = [q.idx for q in bases[0]["base"]]
    indices.append("Host")
    # things that need to match
    indices += ["nCells"]
    df_copy = deepcopy(extra_filter(df))
    df_copy.set_index(keys=indices, inplace=True)
    speedup_df =  eph.helpers.compute_speedup(df_copy, bases, ignore_indices=[]).reset_index()
    return speedup_df[speedup_df["executor"] != "CPU"]



def main(campaign, comparisson=None):
    script_dir = Path(__file__).parent
    post_pro_dir = script_dir / "../postProcessing/{}".format(campaign)
    json_file = post_pro_dir / "results.json"
    df = pd.read_json(json_file)

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



    try:
        unprecond = lambda x: x[x["preconditioner"] == "none"]
        for x, c in [("nCells", "nProcs"), ("nProcs", "nCells")]:
            for y in ["TimeStep", "SolveP"]:
                plotter(
                    x=x,
                    y=y,
                    color=c,
                    style="solver_p",
                    post_pro_dir=post_pro_dir,
                    plot_type="line",
                    col="Host",
                    log=True,
                    df=df,
                    df_filter=Df_filter("unpreconditioned", unprecond),
                )
    except Exception as e:
        print("failed to performe speedup comparisson")
        print(e)

    try:
        speedup = compute_speedup(df, bases)
        speedup.to_json(path_or_buf=post_pro_dir/"speedup_results.json")
        for x, c in [("nCells", "nProcs"), ("nProcs", "nCells")]:
            for y in ["TimeStep", "SolveP"]:
                plotter(
                    x=x,
                    y=y,
                    color=c,
                    style="solver_p",
                    post_pro_dir=post_pro_dir,
                    plot_type="line",
                    col="Host",
                    log=True,
                    df=df,
                    df_filter=Df_filter(
                        "unprecond_speedup", lambda df: compute_speedup(df, bases, unprecond)
                    ),
                )
    except Exception as e:
        print("failed to performe speedup comparisson")
        print(e)

    # comparisson against other results
    try:
        if comparisson:
            for c in comparisson:
                df_orig = df
                post_pro_dir_comp = script_dir / "../postProcessing/{}".format(c)
                json_file = post_pro_dir_comp / "results.json"
                df_comparisson = pd.read_json(json_file)
                df_rel = col_divide(deepcopy(df_orig), deepcopy(df_comparisson))
                print("df_rel", df_rel["TimeStep"])

                for x, c in [("nCells", "nProcs"), ("nProcs", "nCells")]:
                    for y in ["TimeStep", "SolveP"]:
                        plotter(
                            x=x,
                            y=y,
                            color=c,
                            style="solver_p",
                            post_pro_dir=post_pro_dir,
                            postfix="_vs_comparisson",
                            plot_type="line",
                            col="Host",
                            log=True,
                            df=df_rel,
                            df_filter=Df_filter("unpreconditioned", unprecond),
                        )
    except:
        pass


if __name__ == "__main__":
    comparisson = sys.argv[2:] if len(sys.argv) > 2 else None
    main(sys.argv[1], comparisson)
