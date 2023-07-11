import os
import exasim_plot_helpers as eph
from Owls.parser.LogFile import LogFile, LogKey
from pathlib import Path
from subprocess import check_output
from obr.signac_wrapper.operations import JobCache
from obr.OpenFOAM.case import OpenFOAMCase
from copy import deepcopy
import sys


def append_update(d, key, updater):
    prev_res = d.get(key, [])
    prev_res.append(updater())
    d[key] = prev_res


def find_logs(job):
    """Find and return all solver logs files which are of the form *Foam-<TimeStamp>.log"""
    case_path = Path(job.path)
    if not case_path.exists():
        return

    root, campaigns, _ = next(os.walk(case_path))

    def find_tags(path: Path, tags: list, tag_mapping):
        """Recurses into subfolders of path until a system folder is found

        Returns:
          Dictionary mapping paths to tags -> tag
        """
        # TODO implement a more robust way to determine is_case
        _, folder, _ = next(os.walk(path))
        is_case = len(folder) == 0
        if is_case:
            tag_mapping[str(path)] = tags
        else:
            for f in folder:
                tags_copy = deepcopy(tags)
                tags_copy.append(f)
                find_tags(path / f, tags_copy, tag_mapping)
        return tag_mapping

    for campaign in campaigns:
        # check if case folder
        tag_mapping = find_tags(case_path / campaign, [], {})

        for path, tags in tag_mapping.items():
            root, _, files = next(os.walk(path))
            for file in files:
                if "Foam" in file and file.endswith("log"):
                    yield f"{root}/{file}", campaign, tags


def generate_log_keys():
    col_iter = ["init", "final", "iter"]
    col_time = ["time"]
    p_steps = ["_p", "_pFinal"]
    U_components = ["_Ux", "_Uy", "_Uz"]

    pIter = LogKey("Solving for p", col_iter, p_steps)
    UIter = LogKey("Solving for U", col_iter, U_components)
    continuityError = LogKey(
        "time step continuity errors", ["local", "global", "cumulative"]
    )

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
    logKeys += [continuityError]
    return logKeys, combinedKeys, pKeys


def get_timestamp_from_log(log):
    log_name = Path(log).stem
    before = log_name.split("_")[0]
    return log_name.replace(before + "_", "")


def get_OGL_from_log(log):
    try:
        ret = check_output(["grep", "'OGL commit'", log], text=True)
        return ret.split(":")[-1]
    except:
        return "None"


def get_log_from_job(job, campaign, tags, fn, func):
    """Check for cells from blockMeshDict or checkMesh logs"""
    tag = "/".join(tags)
    case_path = Path(job.path) / f"{campaign}/{tag}" 
    print("case_path", case_path)
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


def get_cells_from_job(job, campaign, tags):
    """Check for cells from blockMeshDict or checkMesh logs"""

    def func(file):
        ret = check_output(["grep", "nCells", file], text=True)
        return int(ret.split(":")[-1])

    return get_log_from_job(job, campaign, tags, "blockMesh", func)


def get_sub_domains_from_job(job, campaign, tags):
    def func(file):
        ret = check_output(["grep", "Processor", file], text=True)
        return int(len(ret.split("\n")) / 2)

    return get_log_from_job(job, campaign, tags, "decomposePar", func)


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


        run_logs = []
        logKeys, combinedKeys, pKeys = generate_log_keys()
        # find all solver logs
        for log, campaign, tags in find_logs(job):
            job.doc["obr"]["postprocessing"]["nCells"] = get_cells_from_job(job, campaign, tags)
            job.doc["obr"]["postprocessing"]["decomposition"] = {
                "nSubDomains": get_sub_domains_from_job(job, campaign, tags)
            }
            print("postPro", log, campaign, tags)
            log_file_parser = LogFile(logKeys)
            df = log_file_parser.parse_to_df(log)
            timestamp = get_timestamp_from_log(log)
            record = {
                "timestamp": timestamp,
                "host": log_file_parser.header.host,
                "campaign": campaign,
                "tags": ",".join(tags),
                "OGL_commit": get_OGL_from_log(log),
            }

            # Store the times
            for k in combinedKeys:
                try:
                    record[k] = df.iloc[1:].mean()["time_" + k]
                except Exception as e:
                    print(df)
                    print("failure e", e)
                    

            for k in pKeys:
                try:
                    record[k] = df.iloc[1:].mean()[k]
                except Exception as e:
                    print(df)
                    print(e)

            for k in ["local", "global", "cumulative"]:
                try:
                    record[f"cont_error_{k}"] = df.iloc[-1][k]
                except:
                    print(job.id, df)
                    record[f"cont_error_{k}"] = 0



            run_logs.append(record)

        job.doc["obr"]["postprocessing"]["runs"] = run_logs
