import os

from pathlib import Path
from obr.core.core import find_tags
from signac.job import Job
from shutil import copytree, rmtree

"""
This script will transform the tree structure from:
 └── fd52708db5c296d1fa52b056701be4ee
        ├── campaign1
        │  ├── testTag
        │  │  ├── subtag
        │  │  │  ├── decomposePar_2024-01-05_17:38:44.log
        │  │  │  ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │  │  │  └── solverExitCode.log
        │  │  └── subtag2
        │  │      ├── decomposePar_2024-01-05_17:38:44.log
        │  │      ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │  │      └── solverExitCode.log
        │  ├── testTag2
        │  │  └── subtag3
        │  │      ├── decomposePar_2024-01-05_17:38:44.log
        │  │      ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │  │      └── solverExitCode.log
        │  └── testTag3
        │      ├── decomposePar_2024-01-05_17:38:44.log
        │      ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │      └── solverExitCode.log
        ├── signac_job_document_6850e48aa71863aac82ca3c2cdd6d5e2_campaign1.json
        └── signac_statepoint.json

to:

└── fd52708db5c296d1fa52b056701be4ee
        ├── campaign1
        │  ├── [subtag2][testTag]
        │  │  ├── decomposePar_2024-01-05_17:38:44.log
        │  │  ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │  │  └── solverExitCode.log
        │  ├── [subtag3][testTag2]
        │  │  ├── decomposePar_2024-01-05_17:38:44.log
        │  │  ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │  │  └── solverExitCode.log
        │  ├── [subtag][testTag]
        │  │  ├── decomposePar_2024-01-05_17:38:44.log
        │  │  ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │  │  └── solverExitCode.log
        │  └── [testTag3]
        │      ├── decomposePar_2024-01-05_17:38:44.log
        │      ├── instrumentedPimpleFoam_2024-01-05_17:44:32.log
        │      └── solverExitCode.log
        ├── signac_job_document_6850e48aa71863aac82ca3c2cdd6d5e2_campaign1.json
        └── signac_statepoint.json
"""


def safe_copy_tag(src: str, dst: str, tags=list[str]) -> None:
    sorted_tags = sorted(tags)
    tag_name = f"[{']['.join(sorted_tags)}]"
    tag_path = Path(dst) / tag_name

    if not tag_path.exists():
        tag_path.mkdir()
    print(f"copytree({src, tag_path})")
    copytree(src, tag_path, dirs_exist_ok=True)


def call(jobs: list[Job], kwargs={}) -> None:
    campaign = kwargs.get("campaign", "")
    for job in jobs:
        campaign_path = f"{job.path}/{campaign}"
        if not Path(campaign_path).exists():
            continue
        _, _, files = next(os.walk(campaign_path))

        if len(files) > 1:
            raise AssertionError(f"Fauly state detected for {job.id}. Please remove all loose files in {campaign_path}")

        tag_mapping = find_tags(path=Path(campaign_path), tags=[], tag_mapping={})
        visited = set()
        for tag_path, tags in tag_mapping.items():
            safe_copy_tag(src=tag_path, dst=campaign_path, tags=tags)
            visited.add(tags[0])
        for vis in visited:
            rmtree(path=f"{campaign_path}/{vis}")
