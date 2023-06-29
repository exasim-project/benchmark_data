import os
import exasim_plot_helpers as eph
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
    """Find and return all solver logs files which are of the form *Foam-<TimeStamp>.log"""
    case_path = Path(job.path) / "case"
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
    OGLAnnotationKeys = eph.helpers.build_OGLAnnotationKeys(["p"])

    OGLAnnotations = [
        LogKey(search, ["proc", "time"], append_search_to_col=True)
        for search in OGLAnnotationKeys
    ]

    # Solver annotations
    SolverAnnotationKeys = eph.helpers.SolverAnnotationKeys

    pKeys = [c + p for c in col_iter for p in p_steps]
    UKeys = [c + p for c in col_iter for p in U_components]

    combinedKeys = SolverAnnotationKeys + OGLAnnotationKeys + pKeys  # + UKeys

    SolverAnnotations = [
        LogKey(search, col_time, append_search_to_col=True)
        for search in SolverAnnotationKeys
    ]

    # logKeys = [pIter, UIter]
    logKeys = [pIter]
    logKeys += SolverAnnotations
    logKeys += OGLAnnotations
    return logKeys, combinedKeys, pKeys


def get_timestamp_from_log(log):
    log_name = Path(log).stem
    before = log_name.split("_")[0]
    return log_name.replace(before + "_", "")

def get_log_from_job(job, fn, func):
    """Check for cells from blockMeshDict or checkMesh logs"""
    case_path = Path(job.path) / "case"
    if not case_path.exists():
        return
    root, _, files = next(os.walk(case_path))
    sel_file = None
    for file in files:
        if fn in file and file.endswith("log"):
            sel_file = f"{root}/{file}"

    if sel_file:
        return func(sel_file)
    return

def get_cells_from_job(job):
    """Check for cells from blockMeshDict or checkMesh logs"""
    def func(file):
        ret = check_output(["grep", "nCells", file], text=True)
        return int(ret.split(":")[-1])
    return get_log_from_job(job, "blockMesh", func)


def get_sub_domains_from_job(job):
    def func(file):
        ret = check_output(["grep", "Processor", file], text=True)
        return int(len(ret.split("\n")) / 2)
    return get_log_from_job(job, "decomposePar", func)


def call(jobs):
    """Based on the passed jobs all existing log files are parsed the 
    results get stored in the job document using the following schema 

     obr:
         postprocessing:
           cells:
           decomposition: {}
           ...
           runs: [
             - {host: hostname, timestamp: timestamp, results: ...} <- records
            ]
    """
    cache = JobCache(jobs)

    for job in jobs:
        job.doc["obr"] = {"postprocessing": {}}
        #

        job.doc["obr"]["postprocessing"]["nCells"] = get_cells_from_job(job)
        job.doc["obr"]["postprocessing"]["decomposition"] = {
            "nSubDomains": get_sub_domains_from_job(job)
        }

        run_logs = []
        logKeys, combinedKeys, pKeys = generate_log_keys()
        # find all solver logs
        for log in find_logs(job):
            log_file_parser = LogFile(logKeys)
            df = log_file_parser.parse_to_df(log)
            timestamp = get_timestamp_from_log(log)
            record = {
                "timestamp": timestamp,
                "host": log_file_parser.header.host,
            }

            # Store the times 
            for k in combinedKeys:
                try:
                    record[k] = df.iloc[1:].mean()["time_" + k]
                except:
                    pass

            for k in pKeys:
                try:
                    record[k] = df.iloc[1:].mean()[k]
                except:
                    pass

            run_logs.append(record)

        job.doc["obr"]["postprocessing"]["runs"] = run_logs
