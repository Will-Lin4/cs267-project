#include "common.h"
#include <set>

#define TAG_REDUCE_SCATTER 0
#define TAG_ALL_GATHER 1

// =================
// Helper Functions
// =================

/* Modulus operator without a negative output */
int mod(int a, int b) {
	int res = a%b;
	return res < 0 ? res+b : res;
}

std::map<int, std::set<int>> compute_partition_chunks(const int num_procs, const int vector_len) {
	std::map<int, std::set<int>> chunks;

	const int chunk_size = vector_len / num_procs;
	int idx;
	for (int rank = 0; rank < num_procs; rank++) {
		for (idx = rank*chunk_size; idx < (rank+1)*chunk_size; idx++) {
			chunks[rank].insert(idx);
		}
	}
	for (; idx < vector_len; idx++) { // leftover incides to final processor
		chunks[num_procs-1].insert(idx);
	}

	return chunks;
}

// =================
// Direct Allreduce
// =================

void reduce_scatter(const int num_procs, const int rank, const int vector_len,
					const std::map<int, int>& in_vector, std::map<int, int>& reduced_chunk,
					std::map<int, std::set<int>> chunks) {
	// Initialize reduced_chunk with in_vector
	if (in_vector.size() < chunks[rank].size()) {
		for (const auto& [idx, val] : in_vector) {
			if (chunks[rank].count(idx)) {
				reduced_chunk[idx] = val;
			}
		}
	} else {
		for (auto idx : chunks[rank]) {
			auto pair = in_vector.find(idx);
			if (pair != in_vector.end()) {
				reduced_chunk[pair->first] = pair->second;
			}
		}
	}

	/* Send all other chunks to respective processors
	   and read + reduce the chunks of all other processors. */
	int* recv_data = new int[2*vector_len + 1];
	for (int dist = 1; dist < num_procs; dist++) {
		MPI_Request send_request, recv_request;
		int dst = (rank+dist) % num_procs;
		int src = mod(rank-dist, num_procs);

		// Setup send data: [nnz, idx_1, ..., idx_nnz, val_1, ..., val_nnz]
		int send_nnz = chunks[dst].size();
		int* send_data = new int[2*send_nnz + 1];
		send_data[0] = send_nnz;
		int i = 1;
		if (in_vector.size() < chunks[dst].size()) {
			for (const auto& [idx, val] : in_vector) {
				if (chunks[dst].count(idx)) {
					send_data[i] = idx;
					send_data[i+send_nnz] = val;
					i++;
				}
			}
		} else {
			for (auto idx : chunks[dst]) {
				auto pair = in_vector.find(idx);
				if (pair != in_vector.end()) {
					send_data[i] = pair->first;
					send_data[i+send_nnz] = pair->second;
					i++;
				}
			}
		}

		MPI_Isend(send_data, 2*send_nnz+1, MPI_INT, dst, TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(recv_data, 2*vector_len+1, MPI_INT, src, TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &recv_request);
		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

		int recv_nnz = recv_data[0];
		for (int i = 1; i <= recv_nnz; i++) {
			int idx = recv_data[i];
			int val = recv_data[i+recv_nnz];
			reduced_chunk[idx] += val;
		}

		delete send_data;
	}
	delete recv_data;
}

void all_gather(const int num_procs, const int rank, const int vector_len,
				const std::map<int, int>& reduced_chunk, std::map<int, int>& reduced_vector) {
	// Send reduced_chunk to all other processors
	int send_nnz = reduced_chunk.size();
	int* send_data = new int[2*send_nnz + 1];
	send_data[0] = send_nnz;
	int i = 1;
	for (const auto& [idx, val] : reduced_chunk) {
		reduced_vector[idx] = val;
		send_data[i] = idx;
		send_data[i+send_nnz] = val;
		i++;
	}
	for (int dist = 1; dist < num_procs; dist++) {
		int dst = (rank+dist) % num_procs;
		MPI_Send(send_data, 2*send_nnz+1, MPI_INT, dst, TAG_ALL_GATHER, MPI_COMM_WORLD);
	}
	delete send_data;

	// Read reduced chunks from all other processors
	int* recv_data = new int[2*vector_len + 1];
	for (int dist = 1; dist < num_procs; dist++) {
		int src = mod(rank-dist, num_procs);
		MPI_Recv(recv_data, 2*vector_len+1, MPI_INT, src, TAG_ALL_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int recv_nnz = recv_data[0];
		for (int i = 1; i <= recv_nnz; i++) {
			int idx = recv_data[i];
			int val = recv_data[i+recv_nnz];
			reduced_vector[idx] = val;
		}
	}
	delete recv_data;
}

void sparse_all_reduce(const int num_procs, const int rank, const int vector_len,
					   const std::map<int, int>& in_vector, std::map<int, int>& reduced_vector) {
	std::map<int, std::set<int>> chunks = compute_partition_chunks(num_procs, vector_len);
	std::map<int, int> reduced_chunk;
	reduce_scatter(num_procs, rank, vector_len, in_vector, reduced_chunk, chunks);
	MPI_Barrier(MPI_COMM_WORLD);
	all_gather(num_procs, rank, vector_len, reduced_chunk, reduced_vector);
	MPI_Barrier(MPI_COMM_WORLD);
}
