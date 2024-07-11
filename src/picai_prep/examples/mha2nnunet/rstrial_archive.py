#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import json
import os
from typing import Optional

from picai_prep.data_utils import PathLike
from tqdm import tqdm


def generate_nifti2nnunet_settings(
    archive_dir: PathLike,
    output_path: PathLike,
    annotations_dir: Optional[PathLike] = None,
    task: str = "Task001_rstrial_nnunet_v1",
):
    """
    Create mha2nnunet_settings.json (for inference) for an MHA archive with the following structure:
    /path/to/archive/
    ├── [patient UID]/
        ├── [patient UID]_[study UID]_[modality].mha
        ...

    Parameters:
    - archive_dir: path to MHA archive
    - output_path: path to store MHA -> nnUNet settings JSON to
        (parent folder should exist)
    """
    archive_list = []

    # traverse MHA archive
    for patient_id in tqdm(sorted(os.listdir(archive_dir))):
        # traverse each patient's studies
        patient_dir = os.path.join(archive_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # collect list of available studies
        files = os.listdir(patient_dir)

        # construct scan paths
        scan_paths = [
            f"{patient_id}/{modality}.nii.gz"
            for modality in ["t2", "adc", "highb"]
        ]
        all_scans_found = all([
            os.path.exists(os.path.join(archive_dir, path))
            for path in scan_paths
        ])

        # construct annotation path
        annotation_path = f"Uoa{patient_id}_uncropped_mask_nifti.nii.gz"

        if annotations_dir is not None:
            # check if annotation exists
            if not os.path.exists(os.path.join(annotations_dir, annotation_path)):
                # could not find annotation, skip case
                continue

        if all_scans_found:
            # store info for complete studies
            archive_list += [{
                "patient_id": patient_id,
                "study_id": '1',
                "scan_paths": scan_paths,
                "annotation_path": annotation_path,
            }]

    mha2nnunet_settings = {
        "dataset_json": {
            "task": task,
            "description": "bpMRI scans from PI-CAI dataset to train nnUNet baseline",
            "tensorImageSize": "4D",
            "reference": "",
            "licence": "",
            "release": "1.0",
            "modality": {
                "0": "T2W",
                "1": "CT",
                "2": "HBV"
            },
            "labels": {
                "0": "background",
                "1": "lesion"
            }
        },
        "preprocessing": {
            # optionally, resample and perform centre crop:
            # "matrix_size": [
            #     20,
            #     320,
            #     320
            # ],
            # "spacing": [
            #     3.0,
            #     0.5,
            #     0.5
            # ],
        },
        "archive": archive_list
    }

    if not len(archive_list):
        raise ValueError(f"Did not find any MHA scans in {archive_dir}, aborting.")

    with open(output_path, "w") as fp:
        json.dump(mha2nnunet_settings, fp, indent=4)

    print(f""""
    Saved mha2nnunet_settings to {output_path}, with {len(archive_list)} cases.
    """)
