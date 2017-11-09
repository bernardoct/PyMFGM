#!/bin/bash
#PBS -o output1.out
#PBS -e error1.err
#PBS -l nodes=1:ppn=16
#PBS -l walltime=1:00:00
#PBS -j oe
cd $PBS_O_WORKDIR
time python stress_monosample1.py
