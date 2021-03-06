# Exploiting Heterogeneous Architectures for Rigid Image Registration
This repository contains the code relative the publication "Exploiting Heterogeneous Architectures for Rigid Image Registration" at BioCAS 2021

## Testing Environment
1. We tested the code on linux-based machines (Ubuntu 18.04, CentOS 7), the paper machine is the CentOS one with 4-core Intel i7-6700 and an NVIDIA GTX 1660 Super.
2. We used python 3.6 with `pydicom` `cv2` `numpy` `pandas` `torch` `kronia` `argparse` `statistics` packets on the generation machine
3. Data used in this publication were generated by the National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC).https://doi.org/10.7937/k9/tcia.2018.pat12tbs. Patient: C3N-00704, Study: Dec 10, 2000 NM PET 18 FDG SKULL T, CT: WB STND, PET: WB 3D AC)
4. The 1+1 code takes inspiration from [ITK code](https://github.com/InsightSoftwareConsortium/ITK)

## Code organization
* `*.py` python source code for the 1+1 or Powell's optimizations procedures
* `run_script.sh` automation script to run extensive tests for both CPU and CUDA-based platforms


#### Credits and Contributors

Contributors:  D'Arnese, Eleonora and Del Sozzo, Emanuele and Conficconi, Davide and Santambrogio, Marco D.

If you find this repository useful, please use the following citation(s):

```
@inproceedings{faberbiocas2021,
author = {D'Arnese, Eleonora and Del Sozzo, Emanuele and Conficconi, Davide and Santambrogio, Marco D.},
title = {Exploiting Heterogeneous Architectures for Rigid Image Registration},
booktitle = {2021 IEEE Biomedical Circuits and Systems Conference (BioCAS)},
pages={1--4},
year = {2021},
organization={IEEE}
}
```