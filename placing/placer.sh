#!/bin/bash

dataset=$1
object=$2
scale=$3

if [ "$scale" -eq 0 ]; then
    iteration=15000
else
    iteration=90000
fi

data_path="gaussian-grouping/data/${dataset}"
output_path="gaussian-grouping/data/${dataset}"

select_obj_id='1'

### zooming ###

if [ "$scale" -eq 0 ]; then
    python placer-zooming.py --dataset ${dataset} --object ${object} --iteration ${iteration} --select_obj_id ${select_obj_id}
    zooming_file="tracking/placer-zooming/output.txt"
    loc_coord=$(head -n 1 "$zooming_file")
    supp_obj=$(tail -n 2 "$zooming_file" | head -n 1)
else
    python placer-zooming.py --dataset ${dataset} --object ${object} --iteration ${iteration} --select_obj_id ${select_obj_id}
    zooming_file="tracking/placer-zooming/output.txt"
    loc_coord=$(tail -n 1 "$zooming_file")
    supp_obj=$(tail -n 2 "$zooming_file" | head -n 1)
fi

### locating ###

python placer-locating.py --dataset ${dataset} --object ${object} --iteration ${iteration} --select_obj_id ${select_obj_id} --coord ${loc_coord} --supp_object ${supp_obj}
