#!/usr/bin/bash
set -e

temperature=T2

python train_gmm.py \
	--data=results_ae/encodings/$temperature.npy \
	--epochs=10 \
	--results_dir=results_gmm/$temperature \
	--device=cpu \
