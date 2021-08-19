#!/bin/bash

# /******************************************
# *MIT License
# *
# *Copyright (c) [2021] [Eleonora D'Arnese, Emanuele Del Sozzo, Davide Conficconi,  Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# ******************************************/

           
IMG_DIM=512

PYCODE_Powell=powell_torch.py
PYCODE_oneplusone=one_plus_one_torch.py

DATASET_FLDR=./
CT_PATH=SE0
PET_PATH=SE4
RES_PATH=results

metric=( MI CC MSE )
dev=( cpu cuda )

for i in "${metric[@]}"
do
    for j in "${dev[@]}"
    do

        echo "python $PYCODE_Powell -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j"
        python $PYCODE_Powell -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/powell_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j

        echo "python $PYCODE_oneplusone -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j"
        python $PYCODE_oneplusone -pt 1 -o 0 -cp $CT_PATH -pp $PET_PATH -rp $RES_PATH/oneplusone_${i}_${j} -t 1 -px $DATASET_FLDR -im $IMG_DIM -mtr $i -dvc $j    
    done
done
