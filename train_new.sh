#!/usr/bin/bash
set -e

Js=("ferromagnetic" "antiferromagnetic")
Ls=(16 32 64 128)
Ns=(2 4 8 16 32 64 128 256)

Es=(128 128 64 64 32 32 16 16)
Bs=(8 16 16 32 32 64 64 128)
LRs=(4e-3 4e-3 2e-3 2e-3 1e-3 1e-3 5e-4 5e-4)

fold=$1
seeds=(0 1 2)

for seed in ${seeds[@]}
do
	#
	# baseline-autoencoder
	for J in ${Js[@]}
	do
		for L in ${Ls[@]}
		do
			for i in ${!Ns[@]}
			do
				python train_new2.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=${Es[i]} \
					--lr=${LRs[i]} \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--results_dir=results5/$J/latent/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu
			done
		done
	done
	#
	# GE-autoencoder
	for J in ${Js[@]}
	do
		for L in ${Ls[@]}
		do
			for i in ${!Ns[@]}
			do
				python train_new2.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=${Es[i]} \
					--lr=${LRs[i]} \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--results_dir=results5/$J/latent_equivariant/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
	#
done

echo All done!
