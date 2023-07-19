import os
import exasim_plot_helpers as eph
import sys

from pathlib import Path
from subprocess import check_output
from obr.OpenFOAM.case import OpenFOAMCase
from copy import deepcopy
from Owls.parser.LogFile import LogFile


def get_OGL_from_log(log):
    try:
        ret = check_output(["grep", "'OGL commit'", log], text=True)
        return ret.split(":")[-1]
    except:
        return "None"


def get_cells_from_job(job, campaign, tags):
    """Check for cells from blockMeshDict or checkMesh logs"""

    def func(file):
        ret = check_output(["grep", "nCells", file], text=True)
        return int(ret.split(":")[-1])

    return eph.import_benchmark_data.get_logfile_from_job(
        job, campaign, tags, "blockMesh", func
    )


def get_sub_domains_from_job(job, campaign, tags):
    def func(file):
        ret = check_output(["grep", "Processor", file], text=True)
        return int(len(ret.split("\n")) / 2)

    return eph.import_benchmark_data.get_logfile_from_job(
        job, campaign, tags, "decomposePar", func
    )


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
        job.doc["obr"] = {"postprocessing": {}}

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
        for log, campaign, tags in eph.import_benchmark_data.find_logs(job):
            job.doc["obr"]["postprocessing"]["nCells"] = get_cells_from_job(
                job, campaign, tags
            )
            job.doc["obr"]["postprocessing"]["decomposition"] = {
                "nSubDomains": get_sub_domains_from_job(job, campaign, tags)
            }

            # Base record
            timestamp = eph.import_benchmark_data.get_timestamp_from_log(log)
            record = {
                "timestamp": timestamp,
                "campaign": campaign,
                "tags": ",".join(tags),
                "OGL_commit": get_OGL_from_log(log),
            }

            for log_key_type, log_keys in log_keys_collection.items():
                print("parse", log, log_keys)
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
        job.doc["obr"]["postprocessing"]["runs"] = run_logs
