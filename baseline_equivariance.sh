#!/usr/bin/bash
set -e

Js=("ferromagnetic" "antiferromagnetic")
Ls=(16 32 64 128)
Ns=(8 16 32 64 128 256)

fold=$1
seeds=(0 1 2)


for J in ${Js[@]}
do
	for L in ${Ls[@]}
	do
		for i in ${!Ns[@]}
		do
			for seed in ${seeds[@]}
			do
				# baseline-autoencoder
				python baseline_equivariance.py \
					--data_dir=../phasefinder/data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--load_model=../phasefinder/results4/$J/latent/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--results_dir=results/$J/baseline_equivariance/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu
			done
		done
	done
done

echo All done!
