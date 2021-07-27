#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int rand_spin(void) {
	return 2*rand()%2+1;
}


int rand_bern(double p) {
	if ((double)rand()/(double)RAND_MAX < p) {
		return 1;
	}
	else {
		return 0;
	}
}


int wrap_int(int a, int b) {
	if (a >= 0 && a < b) {
		return a;
	}
	if (a < 0) {
		return b+a;
	}
	if (a >= b) {
		return a-b;
	}
}


int main(int argc, char *argv[]) {
	int l=16;
	int n_sweeps=20005000;
	int meas_interval=500;
	int burnin=5000;
	double T=1.0;

	if (argc==2) {
		T = strtod(argv[1], NULL);
	}

	int k, neighbor_sum, E, DE;
	int lattice[l][l];
	int *neighbors[l][l][4];
	double p;
	double accept[5]={1, 1, 1, exp(-4/T), exp(-8/T)};
	FILE *fp1, *fp2;

	fprintf(stdout, "Initializing lattice . . . \n");
	for (int i=0; i<l; i++) {
		for (int j=0; j<l; j++) {
			lattice[i][j] = rand_spin();
		}
	}

	for (int i=0; i<l; i++) {
		for (int j=0; j<l; j++) {
			k = 0;
			for (int m=-1; m<=1; m+=2) {
				for (int n=-1; n<=1; n+=2) {
					neighbors[i][j][k] = &(lattice[wrap_int(i+m, l)][wrap_int(j+n, l)]);
					k += 1;
				}
			}
		}
	}

	E = 0;
	for (int i=0; i<l; i++) {
		for (int j=0; j<l; j++) {
			neighbor_sum = 0;
			for (int k=0; k<4; k++) {
				neighbor_sum += *(neighbors[i][j][k]);
			}
			E += -neighbor_sum*lattice[i][j];
		}
	}
	E /= 2;

	fp1 = fopen(	"states.txt", "w+");
	fp2 = fopen(	"energies.txt", "w+");

	fprintf(stdout, "Running simulation . . . \n");
	for (int step=0; step<n_sweeps; step++) {

		for (int i=0; i<l; i++) {
			for (int j=0; j<l; j++) {
				neighbor_sum = 0;
				for (int k=0; k<4; k++) {
					neighbor_sum += *(neighbors[i][j][k]);
				}
				DE = 2*neighbor_sum*lattice[i][j];
				p = accept[DE/4+2];
				if (p==1 || rand_bern(p)==1) {
					lattice[i][j] *= -1;
					E += DE;
				}
			}
		}

		if (step>=burnin && step%meas_interval==0) {
			for (int i=0; i<l; i++) {
				for (int j=0; j<l; j++) {
					fprintf(fp1, "%d", (lattice[i][j]+1)/2);
				}
			}
			fprintf(fp1, "\n");
			fprintf(fp2, "%d\n", E);
		}

	}

	fclose(fp1);
	fclose(fp2);
	fprintf(stdout, "Done!\n");

	return 0;
}
