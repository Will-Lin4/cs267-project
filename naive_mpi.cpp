#include "common.h"
#include <mpi.h>
#include <map>
#include <stdlib.h>

void naive_sparse_all_reduce(const int num_procs, const int rank,
							 const int vector_len,
							 const std::map<int, int>& in_vector,
							 int* reduced_vector) {
	int* send = new int[vector_len]();

	for (const auto& elem_it : in_vector) {
		send[elem_it.first] = elem_it.second;
	}

	MPI_Allreduce(send, reduced_vector, vector_len, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	delete send;
}
