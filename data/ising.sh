#!/usr/bin/bash
set -e

temperatures="0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0"

for temperature in $temperatures
do
	./ising $temperature
	rm energies.txt
	mv states.txt isingdata/T$temperature.txt
done
