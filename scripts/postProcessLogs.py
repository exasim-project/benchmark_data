import os
from Owls.parser.LogFile import LogFile, LogKey
from pathlib import Path
from subprocess import check_output
from obr.signac_wrapper.operations import JobCache
from obr.OpenFOAM.case import OpenFOAMCase


def append_update(d, key, updater):
    prev_res = d.get(key, [])
    prev_res.append(updater())
    d[key] = prev_res

def find_logs(job):
    case_path = Path(job.path)/"case"
    if not case_path.exists():
        return

    root, _, files = next(os.walk(case_path))
    for file in files:
        if "Foam" in file and file.endswith("log"):
            yield f"{root}/{file}"
    return

def generate_log_keys():
    col_iter = ["init", "final", "iter"]
    col_time = ["time"]
    p_steps = ["_p", "_pFinal"]
    U_components = ["_Ux", "_Uy", "_Uz"]

    pIter = LogKey("Solving for p", col_iter, p_steps)
    UIter = LogKey("Solving for U", col_iter, U_components)

    # OGL keys
    OGLAnnotationKeys = [
        key.format(field)
        for key in [
            "{}: update_local_matrix_data:",
            "{}: update_non_local_matrix_data:",
            "{}_matrix: call_update:",
            "{}_rhs: call_update:",
            "{}: init_precond:",
            "{}: generate_solver:",
            "{}: solve:",
            "{}: copy_x_back:",
            "{}: solve_multi_gpu",
        ]
        for field in ["Ux", "Uy", "Uz", "p"]
    ]

    OGLAnnotations = [
        LogKey(search, ["proc", "time"], append_search_to_col=True)
        for search in OGLAnnotationKeys
    ]

    # Solver annotations
    SolverAnnotationKeys = [
        "MatrixAssemblyU",
        "MomentumPredictor",
        "SolveP",
        "MatrixAssemblyPI:",
        "MatrixAssemblyPII:",
        "TimeStep",
    ]

    pKeys = [c + p for c in col_iter for p in p_steps]
    UKeys = [c + p for c in col_iter for p in U_components]

    CombinedKeys = SolverAnnotationKeys + OGLAnnotationKeys + pKeys + UKeys

    SolverAnnotations = [
        LogKey(search, col_time, append_search_to_col=True)
        for search in SolverAnnotationKeys
    ]

    # logKeys = [pIter, UIter]
    logKeys = [pIter]
    logKeys += SolverAnnotations
    logKeys += OGLAnnotations
    return logKeys

def get_cells_from_job(job):
    """Check for cells from blockMeshDict or checkMesh logs"""
    case_path = Path(job.path)/"case"
    if not case_path.exists():
        return
    root, _, files = next(os.walk(case_path))
    blockMesh_file = None
    for file in files:
        if "blockMesh" in file and file.endswith("log"):
            blockMesh_file = f"{root}/{file}"

    if blockMesh_file:
        ret = check_output(["grep", "nCells", blockMesh_file],text=True)
        return int(ret.split(":")[-1])
    return

def call(jobs):

    cache = JobCache(jobs)

    for job in jobs:
        job.doc["obr"] = {"postprocessing": {}}
        # obr: 
        #     postprocessing:
        #       cells:
        #       subdomains:
        #       ...
        #       runs: [
        #         - {host: } <- records
        #        ]


        job.doc["obr"]["postprocessing"]["nCells"] = get_cells_from_job(job)
        # find all solver logs
        for log in find_logs(job):
             log_file_parser = LogFile(generate_log_keys())
             df = log_file_parser.parse_to_df(log)




        # solver = job.doc["obr"].get("solver")
        #
        # # skip jobs without a solver
        # if not solver:
        #     continue
        #
        # case_path = Path(job.path) / "case"
        # of_case = OpenFOAMCase(case_path, job)
        #
        # # get latest log
        # runs = job.doc["obr"][solver]
        # solver = job.sp["solver"]
        # # pop old results
        # for k in CombinedKeys:
        #     try:
        #         job.doc["obr"].pop(k)
        #         job.doc["obr"].pop(k + "_rel")
        #     except Exception as e:
        #         print(e)
        #
        # job.doc["obr"]["nCells"] = cache.search_parent(job, "nCells")
        # job.doc["obr"]["nSubDomains"] = int(
        #     of_case.decomposeParDict.get("numberOfSubdomains")
        # )
        #
        # post_pro_dict = {}
        # post_pro_dict["numberRuns"] = len(runs)
        #
        # for i, run in enumerate(runs):
        #     log_path = case_path / run["log"]
        #
        #     # parse logs for given keys
        #     log_file = LogFile(logKeys)
        #     try:
        #         df = log_file.parse_to_df(log_path)
        #     except Exception as e:
        #         print("failed parsing log", log_path, e)
        #         continue
        #
        #     # write information from header
        #     append_update(post_pro_dict, "host", lambda: log_file.header.host)
        #
        #     # write average times to job.doc
        #     for k in CombinedKeys:
        #         try:
        #             append_update(
        #                 post_pro_dict,
        #                 k,
        #                 lambda: df.iloc[1:].mean()["time_" + k],
        #             )
        #         except Exception as e:
        #             print(e)
        #
        #     # write average times step ratio to job.doc
        #     for k, rel in zip(CombinedKeys, ["TimeStep", "Ux: solve_multi_gpu"]):
        #         try:
        #             append_update(
        #                 post_pro_dict,
        #                 k + "_rel",
        #                 lambda: df.iloc[1:].mean()[k + "_rel"],
        #             )
        #         except Exception as e:
        #             print(e)
        #
        #     # write iters to job.doc
        #
        #     # check if run was completed
        #     append_update(post_pro_dict, "completed", lambda: log_file.is_complete)
        #
        # print(post_pro_dict)
        # job.doc["obr"]["postProcessing"] = post_pro_dict