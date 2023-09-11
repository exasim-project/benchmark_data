import os
import exasim_plot_helpers as eph
import sys
import json

from pathlib import Path
from subprocess import check_output
from obr.OpenFOAM.case import OpenFOAMCase
from obr.core.core import merge_job_documents
from copy import deepcopy
from Owls.parser.LogFile import LogFile
from Owls.parser.FoamDict import FileParser
from warnings import warn


def get_OGL_from_log(log):
    try:
        ret = check_output(["grep", "'OGL commit'", log], text=True)
        return ret.split(":")[-1]
    except:
        return "None"


def get_cells_from_cache(job):
    """Check for cells from blockMeshDict or checkMesh logs"""
    root, _, files = next(os.walk(job.path))
    for f in files:
        if not f.startswith("signac_job_document"):
            continue
        with open(Path(root)/f) as fh:
            d = json.load(fh)
            if not d.get("cache"):
                continue
            if d["cache"].get("nCells"):
                return int(d["cache"]["nCells"])
            else:
                continue



def get_sub_domains_from_log(log):
    try:
        print("log", log)
        ret = check_output(["grep", "nProcs", log], text=True)
        print("ret", ret)
        return int(ret.split("\n")[0].split(":")[-1])
    except Exception as e:
        print('exeception',e)
        return "None"

def get_solver_dict(field, inp):
    ret = {}
    ret["solver_" + field] = inp["solvers"][field]["solver"]
    ret["precond_" + field] = inp["solvers"][field].get("preconditioner", "NA")
    ret["smoother_" + field] = inp["solvers"][field].get("smoother", "NA")
    return ret


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
    for job in jobs:
        if not (Path(job.path) / "signac_statepoint.json").exists():
            warn_msg = "invalid job {} found".format(job.id) 
            warn(warn_msg)
            continue
        else:
            print(f"processing job {job.id}")
        job.doc["data"] = []


        run_logs = []
        log_keys_collection = eph.signac_conversion.generate_log_keys()

        # dictionary to keep postpro function which maps from the dataframe
        # and column to a result to keep in the record
        log_key_postpro = {
            # take the mean of all entries in the log file except the first
            "transp_eqn_keys": lambda df, col: df.iloc[1:].mean()[col],
            "foam_annotation_keys": lambda df, col: df.iloc[1:].mean()[col],
            "ogl_annotation_keys": lambda df, col: df.iloc[1:].mean()[col],
            # take only final value
            "cont_error": lambda df, col: df.iloc[-1][col],
        }

        # find all solver logs corresponding to this specific jobs
        for root, log, campaign, tags in eph.import_benchmark_data.find_logs(job):
            fvSolution = FileParser(path=f"{root}/system/fvSolution")

            # Base record
            timestamp = eph.import_benchmark_data.get_timestamp_from_log(log)
            record = {
                "timestamp": timestamp,
                "campaign": campaign,
                "tags": ",".join(tags),
                "OGL_commit": get_OGL_from_log(log),
                "logfile": Path(log).name,
                "nCells": get_cells_from_cache(job),
                "numberOfSubdomains": get_sub_domains_from_log(log)
            }

            record.update(get_solver_dict("p", fvSolution._dict))
            record.update(get_solver_dict("U", fvSolution._dict))

            for log_key_type, log_keys in log_keys_collection.items():
                log_file_parser = LogFile(log_keys)
                df = log_file_parser.parse_to_df(log)
                if df.empty:
                    continue
                record["host"] = log_file_parser.header.host

                for log_key in log_keys:
                    for col in log_key.column_names:
                        if col == "Time":
                            # TODO check how this ends up here
                            continue
                        try:
                            record[col] = log_key_postpro[log_key_type](df, col)
                        except Exception as e:
                            print(
                                f"failure in post processing the Dataframe  col: {col} {str(df.columns)}",
                                e,
                            )
            run_logs.append(record)

        # store all records
        job.doc["data"] = run_logs
