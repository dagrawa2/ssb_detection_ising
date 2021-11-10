#!/usr/bin/bash
set -e

Js=("ferromagnetic" "antiferromagnetic")
Ns=(8 16 32 64 128 256)

Es=(64 64 32 32 16 16)
Bs=(16 32 32 64 64 128)
LRs=(2e-3 2e-3 1e-3 1e-3 5e-4 5e-4)

scales=("16,32" "16,32,64,128")
nscales=(2 4)

fold=$1
seeds=(0 1 2)

# multiscale GE-autoencoder
for seed in ${seeds[@]}
do
	for J in ${Js[@]}
	do
		for i in ${!Ns[@]}
		do
			for j in ${!scales[@]}
			do
				python train_new.py \
					--data_dir=data/$J \
					--Ls=${scales[j]} \
					--L=16 \
					--L_test=16 \
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
					--results_dir=results5/$J/latent_multiscale_${nscales[j]}/L16/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done

# evaluate 2-scale GE-autoencoder on other lattice sizes
Ls=(32 64 128)
for seed in ${seeds[@]}
do
	for J in ${Js[@]}
	do
		for L in ${Ls[@]}
		do
			for i in ${!Ns[@]}
			do
				python train_new.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=2 \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=0 \
					--lr=${LRs[i]} \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--load_model=results5/$J/latent_multiscale_2/L16/N${Ns[i]}/fold$fold/seed$seed \
					--results_dir=results5/$J/latent_multiscale_2/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done

# evaluate 4-scale GE-autoencoder on other lattice sizes
Ls=(32 64 128)
for seed in ${seeds[@]}
do
	for J in ${Js[@]}
	do
		for L in ${Ls[@]}
		do
			for i in ${!Ns[@]}
			do
				python train_new.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=2 \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=0 \
					--lr=${LRs[i]} \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--load_model=results5/$J/latent_multiscale_4/L16/N${Ns[i]}/fold$fold/seed$seed \
					--results_dir=results5/$J/latent_multiscale_4/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done


echo All done!
