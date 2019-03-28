#!/bin/bash
#PBS -o output2.out
#PBS -e error2.err
#PBS -l nodes=1:ppn=16
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -N py_streamflow 

cd $PBS_O_WORKDIR
module load python-2.7.5
time python stress_monosample1.py
