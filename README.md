# PADS: Patient Outcome Assessment and Decision Support

## Acknowledgement

If you use code or concepts available in this repository, we would be grateful if you would cite the PADS paper: https://doi.org/10.3390/jcm14134515


```bibtex
@article{jcm14134515,
author={Pardo, Àlex and Gómez, Josep and Berrueta, Julen and García, Alejandro and Manrique, Sara and Rodríguez, Alejandro and Bodí, María},
title={Combining Predictive Models of Mortality and Time-to-Discharge for Improved Outcome Assessment in Intensive Care Units},
journal={Journal of Clinical Medicine},
volume={14},
year={2025},
number={13},
ARTICLE-number={4515},
url={https://www.mdpi.com/2077-0383/14/13/4515},
pubmedid={40648890},
issn={2077-0383},
doi={10.3390/jcm14134515}
}
```


## Input data

We used Multi-parameter Intelligent Monitoring in Intensive Care (MIMIC-IV) [(Johnson et al. 2023)](https://www.nature.com/articles/s41597-022-01899-x), a freely available data set to develop PADSthe model. The MIMIC-IV data set was collected from the Beth Israel Deaconess Medical Center in Boston, containing more than 70k critically ill patients who were admitted to critical care units. It contains patient demographics, vital signs, records of fluid and medication administration, results of laboratory tests, observations, and notes provided by care professionals. Access to MIMIC-IV is provided via [PhysioNet](https://physionet.org/content/mimiciv/3.1/). The Institutional Review Board at the BIDMC granted a waiver of informed consent and approved the sharing of the research resource.

For this study, we excluded patients with stays shorter than 48 hours (by definition, the proposed models need at least 48 hours worth of data), and those with life support treatment limitation orders.

## Data preprocessing

All the required preprocessing is implemented in the pipeline. It includes:
- NaN imputation (see table below)
- Data normalization

Non-normal values are not checked in the preprocessing phase.

## Variables

The variables used by PADS are the following:


| Variable | Units | Value range | Notes |
| --- | --- | --- | --- |
|stay_id| - | - |The stay ID provided by MIMIC-IV|
|rate_epinephrine| mcg/kg/min | 0 - 4.64 ||
|rate_norepinephrine| mcg/kg/min | 0 - 21.19 ||
|rate_dopamine| mcg/kg/min | 0.2 - 1069.52 ||
|rate_dobutamine| mcg/kg/min | 0.1 - 40.22 ||
|meanbp_min| mmHg | 0.25 - 299 ||
|pao2fio2ratio_novent| - | 8 - 1706 ||
|pao2fio2ratio_vent| - | 1 - 2104 ||
|gcs_min| - | 3 - 15 ||
|bilirubin_max| mg/dL | 0.1 - 87.2 ||
|creatinine_max| mg/dL |  0.1 - 80 ||
|platelet_min| K/uL | 5 - 2360 ||
|admission_age| - |  18 - 102 ||
|charlson_comorbidity_index| - | 0 - 20 ||
|admission_type_Medical| binary | 0 - 1 ||
|admission_type_ScheduledSurgical| binary | 0 - 1 ||
|admission_type_UnscheduledSurgical| binary | 0 - 1 ||
|icu_expire_flag| binary | 0 - 1 ||
|los| days | 2 - 375 | The lower bound is limited due to the requirements of the model|


## How to use the models in inference

Create a virtualenv:

`python -m venv <virtual_env_name>`

Activate the virtualenv:

`<virtual_env_name>\Scripts\activate`

Install required libraries:

`pip install -r code/requirements.txt`

Create a folder for the data:

`mkdir ./data`

Run inference pipeline:

`python code/inference/inference.py`

Refer to [Test custom dataset.ipynb](https://github.com/alex-pardo/paper1/blob/main/Test%20custom%20dataset.ipynb)
