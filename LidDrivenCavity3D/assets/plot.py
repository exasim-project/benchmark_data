import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import exasim_plot_helpers as eph
import logging

from pathlib import Path
from copy import deepcopy
from pathlib import Path


def plot_script():
    return """import pandas as pd

df_json = '{}'
df = pd.DataFrame.read_json(df_json)
    """


def palette():
    return sb.color_palette("tab10")


def plotter(
    x,
    y,
    color,
    style,
    df,
    df_filter,
    post_pro_dir,
    postfix="",
    size=None,
    col=None,
    log="",
    plot_type="line",
):
    df = df_filter(df)
    if df.empty:
        logging.warning("Dataframe empty after filter")
    name = f"{df_filter.name}_{y}_over_{x}_c={color}_s={style}_cols={col}{postfix}_log={log}"
    script_name = name + ".py"

    script_dir = post_pro_dir / "scripts" 
    script_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = post_pro_dir / y
    plot_dir.mkdir(parents=True, exist_ok=True)

    with open(script_dir / script_name, "w") as script:
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
        palette=palette(),
    )
    if log == "both":
        plt.xscale("log")
        plt.yscale("log")
    if log == "x":
        plt.xscale("log")
    fig_name = name + ".png"
    plt.savefig(plot_dir / fig_name)


def col_divide(df_orig, df_comparisson):
    ret = deepcopy(df_orig).set_index("jobid")
    df_orig = df_orig.set_index("jobid")
    df_comparisson = df_comparisson.set_index("jobid")
    for c in df_orig.columns:
        if c == "nCells" or c == "nProcs":
            continue
        try:
            print(
                f"c: {c} ret[c] {df_comparisson[c] / df_orig[c]}  comp:  {df_comparisson[c]} orig: {df_orig[c]}"
            )
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
        return self.func(deepcopy(df))


def compute_speedup(df, bases, extra_filter=lambda df: df, node_based=False):
    # check if bases vals are in df
    bases_clean = []
    for record in bases:
        base = record["base"]
        keep = all([query.val in df[query.idx].values for query in base])
        if keep:
            bases_clean.append(record)
    bases=bases_clean

    # things that need to match
    if node_based:
        indices = [q.idx for q in bases[0]["base"]]
        indices += ["nNodes"]
        exclude = None
    else:
        indices = [q.idx for q in bases[0]["base"]]
        exclude = ["nNodes"]
    indices += ["nCells", "Host"]

    df_copy = deepcopy(extra_filter(df))
    df_copy_set_idx = df_copy.set_index(keys=indices)
    speedup_df = eph.helpers.compute_speedup(
        df_copy_set_idx, bases, ignore_indices=[], exclude=exclude
    ).reset_index()

    return speedup_df[speedup_df["executor"] != "CPU"]


def generate_base(node_based=False):
    """This function generates the list of base cases queries. The base case queries are used to compute the speedup.
    If node_based is set to be true no specific number of nProcs is added to the base case query.

    Returns:
        a list case and base case queries
    """

    # TODO this needs to know nProcs beforehand
    base_ = [
        eph.helpers.DFQuery(idx="preconditioner", val="none"),
        eph.helpers.DFQuery(idx="executor", val="CPU"),
    ]

    base_nla = deepcopy(base_) 
    base_hkn = deepcopy(base_) 
    base_smuc = deepcopy(base_) 

    if not node_based:
        base_nla.append(eph.helpers.DFQuery(idx="nProcs", val=32))
        base_hkn.append(eph.helpers.DFQuery(idx="nProcs", val=76))
        base_smuc.append(eph.helpers.DFQuery(idx="nProcs", val=112))

    return [
        {
            "case": [
                eph.helpers.DFQuery(idx="Host", val="nla"),
            ],
            "base": base_nla,
        },
        {
            "case": [
                eph.helpers.DFQuery(idx="Host", val="hkn"),
            ],
            "base": base_hkn,
        },
        {
            "case": [
                eph.helpers.DFQuery(idx="Host", val="i20"),
            ],
            "base": base_smuc,
        },
    ]


def compute_fvops(df):
    """this function computes fvops"""
    df["fvOpsTimeStep"] = df["nCells"] / df["TimeStep"] * 1000.
    df["fvOpsTimeSolveP"] = df["nCells"] / df["SolveP"] * 1000.
    return df

def compute_fvops_piter(df):
    """this function computes nCellsPerCU"""
    df["fvOpsPIterTimeStep"] = df["nCells"] / df["TimeStep"] * 1000 / df["p_NoIterations"]
    df["fvOpsPIterPSolve"] = df["nCells"] / df["SolveP"] * 1000 / df["p_NoIterations"]
    return df

def compute_nCellsPerCU(df):
    """this function computes nCellsPerCU"""
    df["nCellsPerRank"] = df["nCells"] / df["nProcs"]
    return df


def compute_cloud_cost(df):
    """this function computes nCellsPerCU"""
    df["CostPerHourCloud"] = 0

    def set_compute_cost(df, host, costs):
        executor = costs["executor"]
        cpu_cost = costs["cpu"]
        gpu_cost = costs["gpu"]
        nla_mapping_cpu = np.logical_and(df["Host"] == host,df["executor"] == "CPU") 
        nla_mapping_gpu = np.logical_and(df["Host"] == host,df["executor"] == executor) 
        df.loc[nla_mapping_cpu, "CostPerHourCloud"] = cpu_cost
        df.loc[nla_mapping_gpu, "CostPerHourCloud"] = gpu_cost + cpu_cost

    set_compute_cost(df, "nla", {"executor": "hip", "cpu": 32*0.08,  "gpu": 8*3.4})
    set_compute_cost(df, "hkn", {"executor": "hip", "cpu": 76*0.1, "gpu": 4*3.4})
    set_compute_cost(df, "i20", {"executor": "hip", "cpu": 112*0.11, "gpu": 4*3.4})
    df["CostPerTimeStepCloud"] = (df["CostPerHourCloud"]/3600.) * (df["TimeStep"] / 1000.) 
    return df


def main(campaign, comparisson=None):
    script_dir = Path(__file__).parent
    post_pro_dir = script_dir / "../postProcessing/{}".format(campaign)
    json_file = post_pro_dir / "results.json"
    df = pd.read_json(json_file)

    df = compute_fvops(df)
    df = compute_fvops_piter(df)
    df = compute_nCellsPerCU(df)
    df = compute_cloud_cost(df)

    unprecond = lambda x: x[x["preconditioner"] == "none"]
    for filt in [
        Df_filter("unpreconditioned", unprecond),
        Df_filter(
            "unprecond_speedup",
            lambda df: compute_speedup(df, generate_base(node_based=False), unprecond),
        ),
        Df_filter(
            "unprecond_speedup_nNodes",
            lambda df: compute_speedup(
                df, generate_base(node_based=True), unprecond, node_based=True
            ),
        ),
    ]:
        for x, c in [
            ("nCells", "nProcs"),
            ("nProcs", "nCells"),
            ("nNodes", "nCells"),
            ("nCellsPerRank", "nCells"),
        ]:
            try:
                for log in ["", "both"]:
                    for y in [
                            "TimeStep", "SolveP",
                            "fvOpsTimeStep",
                            "fvOpsTimeSolveP",
                            "fvOpsPIterTimeStep",
                            "fvOpsPIterSolveP",
                            "CostPerTimeStepCloud",
                            ]:
                        plotter(
                            x=x,
                            y=y,
                            color=c,
                            style="solver_p",
                            post_pro_dir=post_pro_dir,
                            plot_type="line",
                            col="Host",
                            log=log,
                            df=df,
                            df_filter=filt,
                        )
            except Exception as e:
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

                for log in ["", "both"]:
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
                                log=log,
                                df=df_rel,
                                df_filter=Df_filter("unpreconditioned", unprecond),
                            )
    except:
        pass


if __name__ == "__main__":
    comparisson = sys.argv[2:] if len(sys.argv) > 2 else None
    main(sys.argv[1], comparisson)
