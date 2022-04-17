#include "common.h"
#include <mpi.h>
#include <map>
#include <stdlib.h>

void naive_sparse_all_reduce(const int num_procs, const int rank,
							 const int vector_len,
							 const std::map<int, int>& in_vector,
							 std::map<int, int>& reduced_vector) {
	int* send = (int*) calloc(vector_len, sizeof(int));
	int* recv = (int*) malloc(vector_len * sizeof(int));

	for (const auto& elem_it : in_vector) {
		send[elem_it.first] = elem_it.second;
	}

	MPI_Allreduce(send, recv, vector_len, MPI::INT, MPI_SUM, MPI_COMM_WORLD);

	for (int i = 0; i < vector_len; i++) {
		if (recv[i] != 0) {
			reduced_vector[i] = recv[i];
		}
	}

	free(send);
	free(recv);
}
