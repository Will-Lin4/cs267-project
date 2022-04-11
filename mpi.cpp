#include "common.h"

// =================
// Helper Functions
// =================

/* Modulus operator without a negative output */
int mod(int a, int b) {
	int res = a%b;
	return res < 0 ? res+b : res;
}

// =================
// Direct Allreduce
// =================
void sparse_all_reduce(const int num_procs, const int rank, const int vector_len,
					   const std::map<int, int>& in_vector, std::map<int, int>& reduced_vector) {
	const int n = vector_len;
	const int nnz = in_vector.size();

	// Copy in_vec to reduced_vec + setup send data
	// [nnz, idx_1, ..., idx_nnz, val_1, ..., val_nnz]
	int* send_data = new int[2*nnz + 1];
	send_data[0] = nnz;
	int i = 1;
	for (const auto& [idx, val] : in_vector) {
		reduced_vector[idx] = val;
		send_data[i] = idx;
		send_data[i+nnz] = val;
		i++;
	}

	int* recv_data = new int[2*n+1]; // TODO: Use MPI_Get_count instead of full 2n+1 for recv_data
	for (int dist = 1; dist < num_procs; dist++) {
		MPI_Request send_request, recv_request;
		int dst = (rank+dist) % num_procs;
		int src = mod(rank-dist, num_procs);
		MPI_Isend(send_data, 2*nnz+1, MPI_INT, dst, 0, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(recv_data, 2*n+1, MPI_INT, src, 0, MPI_COMM_WORLD, &recv_request);
		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

		int recv_nnz = recv_data[0];
		for (int i = 1; i <= recv_nnz; i++) {
			int idx = recv_data[i];
			int val = recv_data[i+recv_nnz];
			reduced_vector[idx] += val;
		}
	}
	delete send_data;
	delete recv_data;

	MPI_Barrier(MPI_COMM_WORLD);
}
