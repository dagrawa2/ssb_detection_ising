#!/usr/bin/bash
set -e

Js=("ferromagnetic" "antiferromagnetic")
Ls=(16 32 64 128)
Ns=(8 16 32 64 128 256)

Es=(64 64 64 64 64 64)
Bs=(8 16 32 64 128 256)
LR=1e-3

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
				# GE-autoencoder
				python train.py \
					--data_dir=../phasefinder/data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=${Es[i]} \
					--lr=$LR \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--results_dir=results/$J/latent_equivariant_2D/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--latent_dim=2 \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done


echo All done!
