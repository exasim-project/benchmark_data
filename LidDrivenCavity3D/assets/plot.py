import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import exasim_plot_helpers as eph
import logging
import warnings

warnings.filterwarnings("ignore")

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
    if df.empty:
        logging.warning("Dataframe empty after filter")
    name = f"{y}_over_{x}_c={color}_s={style}_cols={col}{postfix}_log={log}"
    script_name = name + ".py"

    script_dir = post_pro_dir / df_filter.name / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = post_pro_dir / df_filter.name / y
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
    # dont normalize nCells, nProcs, or deviceRankOverSubscription
    for c in df_orig.columns:
        if c == "nCells" or c == "nProcs" or c == "deviceRankOverSubscription":
            continue
        try:
            ret[c] = df_comparisson[c] / df_orig[c]
        except Exception as e:
            print(e, c)
            pass
    return ret


class Df_filter:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, df):
        print(f"call {self.name}")
        return self.func(deepcopy(df))


def compute_speedup(df, bases, extra_filter=lambda df: df, node_based=False):
    # df = df[df["Host"] != "nla"] 
    df = extra_filter(df)

    # check if bases vals are actually in df
    # this is required since the basis values are used to compute the speedup
    # if the basis values are not present computing speedup will fail
    bases_clean = []
    for record in bases:
        base = record["base"]
        keep = all([query.val in df[query.idx].values for query in base])
        if keep:
            bases_clean.append(record)
    if not bases_clean:
        error = (f"failed generating clean bases {bases} for {df.to_string()}")
        raise AssertionError(error)
    bases = bases_clean

    # extra things that need to match when doing the division
    if node_based:
        indices = [q.idx for q in bases[0]["base"]]
        indices += ["nNodes"]
        exclude = None
    else:
        indices = [q.idx for q in bases[0]["base"]]
        exclude = ["nNodes"]
    indices += ["nCells", "Host"]

    df_copy_set_idx = df.set_index(keys=indices)
    speedup_df = eph.helpers.compute_speedup(
            df_copy_set_idx, bases, ignore_indices=[], exclude=exclude
        ).reset_index()

    if speedup_df.empty:
        print(f"Computing speedup produced dataframe: Df in {df_copy_set_idx}, bases: {bases}, exclude={exclude}")

    return speedup_df[speedup_df["executor"] != "CPU"]


def generate_base(node_based=False):
    """This function generates the list of base cases queries. The base case queries are used to compute the speedup.
    If node_based is set to be true no specific number of nProcs is added to the base case query.

    Returns:
        a list case and base case queries
    """

    base_ = [
        eph.helpers.DFQuery(idx="preconditioner", val="none"),
        eph.helpers.DFQuery(idx="executor", val="CPU"),
    ]

    base_nla = deepcopy(base_)
    base_hkn = deepcopy(base_)
    base_smuc = deepcopy(base_)

    if not node_based:
        #base_nla.append(eph.helpers.DFQuery(idx="nProcs", val=32))
        base_hkn.append(eph.helpers.DFQuery(idx="nProcs", val=76))
        base_smuc.append(eph.helpers.DFQuery(idx="nProcs", val=112))

    case_hkn = [eph.helpers.DFQuery(idx="Host", val="hkn")] 
    case_smuc = [eph.helpers.DFQuery(idx="Host", val="i20")] 

    # to compute the speedup per node consider the selected  case has  with 2CPUs per GPU
    if node_based:
        case_hkn.append(eph.helpers.DFQuery(idx="deviceRankOverSubscription", val=2))
        case_smuc.append(eph.helpers.DFQuery(idx="deviceRankOverSubscription", val=2))

    return [
       #{
       #    "case": [
       #        eph.helpers.DFQuery(idx="Host", val="nla"),
       #    ],
       #    "base": base_nla,
       #},
        {
            "case": case_hkn, 
            "base": base_hkn,
        },
        {
            "case": case_smuc, 
            "base": base_smuc,
        },
    ]


def compute_fvops(df):
    """this function computes fvops"""
    df["fvOpsTimeStep"] = df["nCells"] / df["TimeStep"] * 1000.0
    df["fvOpsSolveP"] = df["nCells"] / df["SolveP"] * 1000.0
    return df


def compute_fvops_piter(df):
    """this function computes nCellsPerCU"""
    df["fvOpsPIterTimeStep"] = (
        (df["nCells"] * df["p_NoIterations"]) / df["TimeStep"] * 1000
    )
    df["fvOpsPIterSolveP"] = (df["nCells"] * df["p_NoIterations"]) / df["SolveP"] * 1000
    return df


def compute_nCellsPerCU(df):
    """this function computes nCellsPerCU"""
    df["nCellsPerRank"] = df["nCells"] / df["nProcs"]
    return df


def compute_cloud_cost(df):
    """this function computes the cost to compute a timestep based on aws costs"""
    df["CostPerHourCloud"] = 0

    def set_compute_cost(df, host, costs):
        executor = costs["executor"]
        cpu_cost = costs["cpu"]
        gpu_cost = costs["gpu"]
        mapping_cpu = np.logical_and(df["Host"] == host, df["executor"] == "CPU")
        mapping_gpu = np.logical_and(df["Host"] == host, df["executor"] == executor)
        df.loc[mapping_cpu, "CostPerHourCloud"] = cpu_cost
        df.loc[mapping_gpu, "CostPerHourCloud"] = gpu_cost + cpu_cost

    # set_compute_cost(df, "nla", {"executor": "hip", "cpu": 32 * 0.08, "gpu": 8 * 3.4})
    set_compute_cost(df, "hkn", {"executor": "cuda", "cpu": 76 * 0.1, "gpu": 4 * 3.4})
    set_compute_cost(
        df, "i20", {"executor": "dpcpp", "cpu": 112 * 0.11, "gpu": 4 * 3.4}
    )
    df["CostPerTimeStepCloud"] = (df["CostPerHourCloud"] / 3600.0) * (
        df["TimeStep"] / 1000.0
    )
    return df


def compute_gpu_mapping(df):
    """this function computes the nCPU/nGPU mapping"""
    df["deviceRankOverSubscription"] = 0

    def set_compute_cost(df, host, costs):
        executor = costs["executor"]
        cpus = costs["cpu"]
        gpus = costs["gpu"]
        mapping_cpu = np.logical_and(df["Host"] == host, df["executor"] == "CPU")
        mapping_gpu = np.logical_and(df["Host"] == host, df["executor"] == executor)
        df.loc[mapping_cpu, "deviceRanks"] = cpus
        df.loc[mapping_gpu, "deviceRanks"] = gpus

    # set_compute_cost(df, "nla", {"executor": "hip", "cpu": 32, "gpu": 8})
    set_compute_cost(df, "hkn", {"executor": "cuda", "cpu": 76, "gpu": 4})
    set_compute_cost(df, "i20", {"executor": "dpcpp", "cpu": 112, "gpu": 4})
    df["deviceRankOverSubscription"] = df["nProcs"] / df["deviceRanks"]
    return df


def unprecond_rank_range(df):
    mapping = np.logical_and(
        df["preconditioner"] == "none",
        df["deviceRankOverSubscription"] >= 0.9,
    )

    df = df[mapping]
    return df[df["deviceRankOverSubscription"] < 10]


def main(campaign, comparisson=None):
    script_dir = Path(__file__).parent
    post_pro_dir = script_dir / "../postProcessing/{}".format(campaign)
    json_file = post_pro_dir / "results.json"
    df = pd.read_json(json_file)

    df = compute_fvops(df)
    df = compute_fvops_piter(df)
    df = compute_nCellsPerCU(df)
    df = compute_cloud_cost(df)
    df = compute_gpu_mapping(df)

    unprecond = lambda x: x[x["preconditioner"] == "none"]
    for filt in [
        Df_filter("unpreconditioned", unprecond_rank_range),
        Df_filter(
            "unpreconditioned/speedup",
            func = lambda df: compute_speedup(df, generate_base(node_based=False), extra_filter = unprecond),
        ),
        Df_filter(
            "unpreconditioned_rank_range/speedup",
            func = lambda df: compute_speedup(
                df, generate_base(node_based=False), extra_filter =  unprecond_rank_range
            ),
        ),
        Df_filter(
            "unpreconditioned/speedup_nNodes",
            func =lambda df: compute_speedup(
                df, generate_base(node_based=True), extra_filter = unprecond, node_based=True
            ),
        ),
    ]:
        df = filt(df)
        for x, c, h in [
            ("nCells", "nProcs", "Host"),
            ("nProcs", "nCells", "Host"),
            ("nNodes", "nCells", "Host"),
            ("nCells", "deviceRankOverSubscription", "Host"),
            ("deviceRankOverSubscription", "nCells", "Host"),
            ("nCellsPerRank", "nCells", "Host"),
            ("nCells", "Host", "solver_p"),
        ]:
            for log in ["", "x", "both"]:
                for y in [
                    "TimeStep",
                    "SolveP",
                    "fvOpsTimeStep",
                    "fvOpsSolveP",
                    "fvOpsPIterTimeStep",
                    "fvOpsPIterSolveP",
                    "CostPerTimeStepCloud",
                ]:
                    try:
                        plotter(
                            x=x,
                            y=y,
                            color=c,
                            style="solver_p",
                            post_pro_dir=post_pro_dir,
                            plot_type="line",
                            col=h,
                            log=log,
                            df=df,
                            df_filter=filt,
                        )
                    except Exception as e:
                        print(f"Failure plotting filter {filt.name} x:{x}, y:{y}, h:{h}", e)

    # comparisson against other results
    try:
        if comparisson:
            for c in comparisson:
                df_orig = df
                post_pro_dir_comp = script_dir / "../postProcessing/{}".format(c)
                json_file = post_pro_dir_comp / "results.json"
                df_comparisson = pd.read_json(json_file)
                df_rel = col_divide(deepcopy(df_orig), deepcopy(df_comparisson))

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
