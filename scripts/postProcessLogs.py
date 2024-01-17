import os
import sys
import json

from pathlib import Path
from subprocess import check_output
from obr.OpenFOAM.case import OpenFOAMCase
from obr.core.core import merge_job_documents, find_solver_logs, get_timestamp_from_log
from copy import deepcopy
from Owls.parser.LogFile import LogFile, transportEqn, customMatcher
from Owls.parser.FoamDict import FileParser
import pandas as pd


def get_OGL_from_log(log):
    try:
        ret = check_output(["grep", "'OGL commit'", log], text=True)
        return ret.split(":")[-1]
    except:
        return "None"


def Info_Log(name):
    """A wrapper function to create LOG entry parser for the annotated solver"""
    return customMatcher(name, rf"\[INFO\] {name}: (?P<{name}>[0-9.]*) \[ms\]")


def OGL_Log(field, name):
    """A wrapper function to create OGL LOG entry parser"""
    return customMatcher(
        name,
        rf"\[OGL LOG\]\[Proc: 0\]{field}: {name}: (?P<{field + '_' + name}>[0-9.]*) \[ms\]",
    )


def generate_log_keys():
    transport_eqn_keys = [
        transportEqn("Ux"),
        transportEqn("Uy"),
        transportEqn("Uz"),
        transportEqn("p"),
    ]

    ogl_annotation_keys = [
        OGL_Log("p", "update_local_matrix_data"),
        OGL_Log("p", "update_non_local_matrix_data"),
        OGL_Log("p_matrix", "call_update"),
        OGL_Log("p_rhs", "call_update"),
        OGL_Log("p", "solve"),
        OGL_Log("p", "copy_x_back"),
    ]

    foam_annotation_keys = [
        Info_Log("MomentumPredictor"),
        Info_Log("MatrixAssemblyPI"),
        Info_Log("MatrixAssemblyPII"),
        Info_Log("SolveP"),
        Info_Log("PISOStep"),
        Info_Log("TimeStep"),
    ]
    return {
        "transp_eqn_keys": transport_eqn_keys,
        "ogl_annotation_keys": ogl_annotation_keys,
        "foam_annotation_keys": foam_annotation_keys,
    }


def convert_to_numbers(df):
    """convert all columns to float if they dont have Name in it"""
    return df.astype({col: "float" for col in df.columns if not "Name" in col})


def get_average(df, col):
    """comput averages of a column if non a Name column"""
    if "Name" in col:
        return df.iloc[0][col]
    else:
        return df.iloc[1:][col].mean()


def call(jobs, kwargs={}):
    """Based on the passed jobs all existing log files are parsed the
    results get stored in the job document using the following schema

     data:
        [
        # several data records
        {
           campaign: campaign
           id: id
        },
        ]
    """
    campaign = kwargs.get("campaign", "")
    for job in jobs:
        run_logs = job.doc.get("data", [])

        # dictionary to keep postpro function which maps from the dataframe
        # and column to a result to keep in the record
        log_key_postpro = {
            # take the mean of all entries in the log file except the first
            "transp_eqn_keys": get_average,
            "foam_annotation_keys": get_average,
            "ogl_annotation_keys": get_average,
        }

        merge_job_documents(job, str(campaign))

        # find all solver logs corresponding to this specific jobs
        for log, campaign, tags in find_solver_logs(job, campaign):
            # Base record
            log_path = Path(log)
            if use_fvs:=Path(log_path.parent / "system/fvSolution").exists():
                fvSolution = FileParser(
                    path=log_path.parent / "system/fvSolution"
                )
            timestamp = get_timestamp_from_log(log_path)
            record = {
                "timestamp": timestamp,
                "campaign": campaign,
                "tags": ",".join(tags),
                "OGL_commit": get_OGL_from_log(log),
                "logfile": log_path.name,
                "nCells": int(job.doc["cache"].get("nCells", "")),
            }

            for log_key_type, log_keys in generate_log_keys().items():
                log_file_parser = LogFile(log, log_keys)
                df = convert_to_numbers(log_file_parser.parse_to_df())
                record["Host"] = log_file_parser.header.Host[0:3]
                record["nProcs"] = int(log_file_parser.header.nProcs)
                if use_fvs:
                    record["solver_p"] = fvSolution.get("solvers")["p"]["solver"]

                for log_key in log_keys:
                    for col in df.columns:
                        if col == "Time":
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
