#!/usr/bin/bash
set -e

python train.py \
	--data_1=data/T1.txt \
	--data_2=data/T2.txt \
	--batch_size=16 \
	--epochs=100 \
	--lr=2e-2 \
	--reg_weight=0.1 \
	--results_dir=results \
	--device=cpu \
