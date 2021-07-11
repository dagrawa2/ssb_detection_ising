#!/usr/bin/bash
set -e

temperatures="0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0"

for temperature in $temperatures
do
	python train_gmm.py \
		--data=results/AE/encodings/T$temperature.npy \
		--epochs=5 \
		--results_dir=results/GMM/T$temperature \
		--device=cpu \
		--jitter=1e-4 \
		--full_cov
done
