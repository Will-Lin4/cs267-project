#include "common.h"
#include <mpi.h>
#include <map>
#include <stdlib.h>

void naive_sparse_all_reduce(const int num_procs, const int rank,
							 const int vector_len,
							 const std::map<int, int>& in_vector,
							 std::map<int, int>& reduced_vector) {
	int* send = new int[vector_len]();
	int* recv = new int[vector_len];

	for (const auto& elem_it : in_vector) {
		send[elem_it.first] = elem_it.second;
	}

	MPI_Allreduce(send, recv, vector_len, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	for (int i = 0; i < vector_len; i++) {
		if (recv[i] != 0) {
			reduced_vector.emplace(i, recv[i]);
		}
	}

	delete send;
	delete recv;
}