#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 evaluate.py --rec-path ./data/OCCLUSION/val.rec --network resnet50m --batch-size 64 --epoch 45 --data-shape 300 --class-names 'obj_01, obj_02, obj_05, obj_06, obj_08, obj_09, obj_11, obj_12' --prefix ./output/OCCLUSION/resnet50m-300-lr0.001-alpha10-wd0.0005/ssd --gpu 0 --num-class 8


