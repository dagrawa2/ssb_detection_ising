import time
import numpy as np

from phasefinder.datasets import Ising


time_start = time.time()

Ls = [16, 32, 64, 128]
Ts = [1.04+0.04*i for i in range(25)] \
	+ [2.01+0.01*i for i in range(50)] \
	+ [2.54+0.04*i for i in range(25)]

for L in Ls:
	for T in Ts:
		I = Ising()
		I.generate(d=2, 
			L=L, 
			T=T, 
			mc_steps=50000, 
			ieq_steps=10000, 
			meas_steps=10, 
			dynamic="wolff", 
			seed=107, 
			print_state=True, 
			output_dir="data/L{:d}/T{:.2f}".format(L, T), 
			encode=True)

print("Done!")
print("Took {:.3f} seconds.".format(time.time()-time_start))
