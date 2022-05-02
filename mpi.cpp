#include "common.h"
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <boost/math/distributions/geometric.hpp>
#include <boost/math/distributions/poisson.hpp>

#define TAG_REDUCE_SCATTER 0
#define TAG_ALL_GATHER 1
#define TAG_RECURSIVE_DOUBLE 2
#define EPSILON 0

// =================
// Helper Functions
// =================

/* Partitions {0,...,vector_len-1} into num_active_procs subsets of roughly equal
   sparsity based on averaging distribution of nonzero elements across processors */
std::vector<int> estimate_partition_boundaries(const int num_procs, const int vector_len,
											   const std::map<int, int>& in_vector) {
	int num_active_procs = num_procs;
	if (num_active_procs > in_vector.size())
		num_active_procs = in_vector.size();

	std::vector<int64_t> local_boundaries;
	local_boundaries.reserve(num_active_procs + 1);
	local_boundaries.emplace_back(0);

	double chunk_size = (double)in_vector.size() / (double)num_active_procs;

	int nnz = 0;
	for (const auto& elem : in_vector) {
		if (nnz >= std::round(chunk_size * local_boundaries.size())) {
			local_boundaries.emplace_back(elem.first);
		}
		nnz++;
	}

	if (local_boundaries.back() < vector_len)
		local_boundaries.emplace_back(vector_len);

	int64_t* recv_buf = new int64_t[local_boundaries.size()];
	MPI_Allreduce(local_boundaries.data(), recv_buf, local_boundaries.size(),
				  MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

	std::vector<int> avg_boundaries;
	avg_boundaries.reserve(local_boundaries.size());
	for (int i = 0; i < local_boundaries.size(); i++) {
		double avg = double(recv_buf[i]) / num_procs;
		avg_boundaries.emplace_back(avg + 0.5);
	}

	return std::move(avg_boundaries);
}

/* Probability mass function for various distributions */
double pmf (const int x, const char* distribution, const double dist_param) {
	if (!strcmp(distribution, "uniform")) {
		return 1/(double)dist_param;
	} else if (!strcmp(distribution, "geometric")) {
		boost::math::geometric_distribution<> geometric(dist_param);
		return boost::math::pdf(geometric, x);
	} else if (!strcmp(distribution, "poisson")) {
		boost::math::poisson_distribution<> poisson(dist_param);
		return boost::math::pdf(poisson, x);
	}

	return (double)-1;
}

/* Partitions {0,...,vector_len-1} into num_active_procs subsets of roughly equal mass
   according to the probability mass function specified by the distribution */
std::vector<int> compute_partition_boundaries(const int num_procs, const int vector_len,
											  const char* distribution, const double dist_param) {
	std::vector<int> chunk_boundaries; // chunks[i] is first index of processor i's chunk
	if (!strcmp(distribution, "uniform")) {
		int num_active_procs = (num_procs > vector_len) ? vector_len : num_procs;
		double chunk_size = (double)vector_len / (double)num_active_procs;
		for (int i = 0; i <= num_active_procs; i++) {
			int idx = std::round(i * chunk_size);
			chunk_boundaries.push_back(idx);
		}
	} else {
		double mu;
		if (!strcmp(distribution, "geometric")) {
			mu = (1 - dist_param) / dist_param;
		} else if (!strcmp(distribution, "poisson")) {
			mu = dist_param;
		}

		double total_mass = 0;
		for (int i = 0; i < vector_len; i++) {
			total_mass += pmf(i, distribution, dist_param);
			if (total_mass > 0.99) {
				total_mass = 1;
				break;
			}
		}

		int rank = 1;
		double mass = 0;
		const double uniform_mass = 1/(double)num_procs;
		chunk_boundaries.push_back(0);
		for (int idx = 0; idx < vector_len - 1; idx++) {
			/* Assign indices to chunk until cumulative
			probability mass is at least uniform mass */
			double dmass = pmf(idx, distribution, dist_param) / total_mass;
			mass += dmass;
			if (mass >= uniform_mass - EPSILON) {
				chunk_boundaries.push_back(idx + 1);
				rank++;
				mass = 0;
			}

			if (rank == num_procs)
				break; // stop after last processor's chunk is determined

			if (idx > mu && dmass < EPSILON)
				break; // stop prematurely if at tail end of distribution
		}
		chunk_boundaries.push_back(vector_len); // for convenience, so chunks[i+1] always exists
	}

	return std::move(chunk_boundaries);
}


// =========================
// Rabenseifner's Algorithm
// =========================

std::map<int, int> reduce_scatter(const int num_procs, const int num_active_procs,
								  const int my_rank, const int vector_len,
								  const std::map<int, int>& in_vector,
								  const std::vector<int> chunk_boundaries) {
	std::vector<MPI_Request> all_requests;
	all_requests.reserve(num_active_procs + num_procs); // Both send and recv

	// Scatter: For each active processor, send it the relevant chunk of in_vector
	std::vector<std::vector<int>> all_send_data;
	all_send_data.reserve(num_active_procs);

	for (int rank = 0; rank < num_active_procs; rank++) {
		if (rank == my_rank)
			continue;

		// Setup send data: [idx_1, val_1, ..., idx_nnz, val_nnz]
		all_send_data.emplace_back();
		auto& send_data = all_send_data.back();

		for (auto it = in_vector.upper_bound(chunk_boundaries[rank] - 1);
				it != in_vector.lower_bound(chunk_boundaries[rank + 1]); it++) {
			send_data.push_back(it->first);
			send_data.push_back(it->second);
		}

		all_requests.emplace_back();
		MPI_Isend(send_data.data(), send_data.size(), MPI_INT, rank,
				  TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &all_requests.back());
	}

	size_t num_send = all_requests.size();

	// Reduce: Receive and reduce all chunks from other processors
	std::map<int, int> reduced_chunk;

	size_t my_chunk_size = 0;
	int* recv_buffer = nullptr;
	size_t num_recv = 0;

	if (my_rank < num_active_procs) {
		// Initialize reduced_chunk with in_vector
		for (auto it = in_vector.upper_bound(chunk_boundaries[my_rank] - 1);
				it != in_vector.lower_bound(chunk_boundaries[my_rank + 1]); it++) {
			reduced_chunk.emplace(it->first, it->second);
		}

		my_chunk_size = chunk_boundaries[my_rank + 1] - chunk_boundaries[my_rank];
		recv_buffer = new int[2 * my_chunk_size * (num_procs - 1)];

		// Receive chunks from remote processors.
		for (int rank = 0; rank < num_procs; rank++) {
			if (rank == my_rank)
				continue;

			all_requests.emplace_back();
			MPI_Irecv(recv_buffer + num_recv * my_chunk_size * 2, my_chunk_size * 2,
					 MPI_INT, rank, TAG_REDUCE_SCATTER, MPI_COMM_WORLD,
					 &all_requests.back());
			num_recv += 1;
		}
	}

	int index = -1;
	MPI_Status status;
	for (int i = 0; i < all_requests.size(); i++) {
		MPI_Waitany(all_requests.size(), all_requests.data(), &index, &status);

		if (index >= num_send) {
			int recv_size;
			MPI_Get_count(&status, MPI_INT, &recv_size);

			for (int i = 0; i < recv_size; i += 2) {
				int idx = recv_buffer[(index - num_send) * my_chunk_size * 2 + i];
				int val = recv_buffer[(index - num_send) * my_chunk_size * 2 + i + 1];

				reduced_chunk.emplace(idx, 0);
				reduced_chunk[idx] += val;
			}
		}
	}

	if (recv_buffer) {
		delete recv_buffer;
	}

	return std::move(reduced_chunk);
}

void sparse_all_gather(const int num_procs, const int num_active_procs,
					   const int my_rank, const int vector_len,
					   const std::map<int, int>& reduced_chunk,
					   const std::vector<int>& chunk_boundaries,
					   int* reduced_vector) {
	std::vector<MPI_Request> all_requests;
	all_requests.reserve(num_active_procs * 2); // Both send and recv

	// Send reduced_chunk to all other processors
	std::vector<int> send_data;
	if (my_rank < num_active_procs) {
		send_data.reserve(2 * reduced_chunk.size());

		int next_index = chunk_boundaries[my_rank];
		for (const auto& elem : reduced_chunk) {
			send_data.push_back(elem.first);
			send_data.push_back(elem.second);

			for (int i = next_index; i < elem.first; i++) {
				reduced_vector[i] = 0;
			}
			reduced_vector[elem.first] = elem.second;
			next_index = elem.first + 1;
		}
		for (int i = next_index; i < chunk_boundaries[my_rank + 1]; i++) {
			reduced_vector[i] = 0;
		}

		for (int rank = 0; rank < num_procs; rank++) {
			if (rank == my_rank)
				continue;

			all_requests.emplace_back();
			MPI_Isend(send_data.data(), send_data.size(), MPI_INT, rank,
					  TAG_ALL_GATHER, MPI_COMM_WORLD, &all_requests.back());
		}
	}

	size_t num_send = all_requests.size();

	// Read reduced chunks from all active processors
	int* recv_buffer = new int[2 * vector_len];
	for (int rank = 0; rank < num_active_procs; rank++) {
		if (rank == my_rank) {
			continue;
		}

		size_t chunk_size = chunk_boundaries[rank + 1] - chunk_boundaries[rank];

		all_requests.emplace_back();
		MPI_Irecv(recv_buffer + chunk_boundaries[rank] * 2, 2 * chunk_size,
				  MPI_INT, rank, TAG_ALL_GATHER, MPI_COMM_WORLD,
				  &all_requests.back());
	}

	int index = -1;
	MPI_Status status;
	for (int i = 0; i < all_requests.size(); i++) {
		MPI_Waitany(all_requests.size(), all_requests.data(), &index, &status);

		if (index >= num_send) {
			int recv_size;
			MPI_Get_count(&status, MPI_INT, &recv_size);

			int rank = index - num_send;
			if (rank >= my_rank)
				rank += 1;

			int vector_index = chunk_boundaries[rank] * 2;
			int next_index = chunk_boundaries[rank];

			for (int i = 0; i < recv_size; i += 2) {
				int idx = recv_buffer[vector_index + i];
				int val = recv_buffer[vector_index + i + 1];

				for (int j = next_index; j < idx; j++) {
					reduced_vector[j] = 0;
				}

				reduced_vector[idx] = val;
				next_index = idx + 1;
			}

			for (int i = next_index; i < chunk_boundaries[rank + 1]; i++) {
				reduced_vector[i] = 0;
			}
		}
	}

	delete recv_buffer;
}

void dense_all_gather(const int num_procs, const int num_active_procs,
					  const int my_rank, const int vector_len,
					  const std::map<int, int>& reduced_chunk,
					  const std::vector<int>& chunk_boundaries,
					  int* reduced_vector) {
	std::vector<MPI_Request> all_requests;
	all_requests.reserve(num_active_procs * 2); // Both send and recv

	// Send reduced_chunk to all other processors
	int* send_buffer = nullptr;
	if (my_rank < num_active_procs) {
		size_t my_chunk_size =
			chunk_boundaries[my_rank + 1] - chunk_boundaries[my_rank];
		send_buffer = new int[my_chunk_size];

		int next_index = chunk_boundaries[my_rank];
		for (const auto& elem : reduced_chunk) {
			for (int i = next_index; i < elem.first; i++) {
				send_buffer[i - chunk_boundaries[my_rank]] = 0;
				reduced_vector[i] = 0;
			}

			send_buffer[elem.first - chunk_boundaries[my_rank]] = elem.second;
			reduced_vector[elem.first] = elem.second;
			next_index = elem.first + 1;
		}

		for (int i = next_index; i < chunk_boundaries[my_rank + 1]; i++) {
			send_buffer[i - chunk_boundaries[my_rank]] = 0;
			reduced_vector[i] = 0;
		}

		for (int rank = 0; rank < num_procs; rank++) {
			if (rank == my_rank)
				continue;

			all_requests.emplace_back();
			MPI_Isend(send_buffer, my_chunk_size, MPI_INT, rank,
					  TAG_ALL_GATHER, MPI_COMM_WORLD, &all_requests.back());
		}
	}

	size_t num_send = all_requests.size();

	// Read reduced chunks from all active processors
	for (int rank = 0; rank < num_active_procs; rank++) {
		if (rank == my_rank) {
			continue;
		}

		size_t chunk_size = chunk_boundaries[rank + 1] - chunk_boundaries[rank];

		all_requests.emplace_back();
		MPI_Irecv(reduced_vector + chunk_boundaries[rank], chunk_size,
				  MPI_INT, rank, TAG_ALL_GATHER, MPI_COMM_WORLD,
				  &all_requests.back());
	}

	MPI_Waitall(all_requests.size(), all_requests.data(), MPI_STATUSES_IGNORE);

	if (send_buffer)
		delete send_buffer;
}

void dynamic_all_gather(const int num_procs, const int num_active_procs,
					    const int my_rank, const int vector_len,
					    const std::map<int, int>& reduced_chunk,
					    const std::vector<int>& chunk_boundaries,
					    int* reduced_vector) {
	int do_dense_allgather = 0;
	if (my_rank < num_active_procs) {
		size_t my_chunk_size =
			chunk_boundaries[my_rank + 1] - chunk_boundaries[my_rank];

		if (reduced_chunk.size() * 2 > my_chunk_size) {
			do_dense_allgather += 1;
		}
	}

	int total_dense_allgathers = 0;
	MPI_Allreduce(&do_dense_allgather, &total_dense_allgathers,
				  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	if (total_dense_allgathers == num_active_procs) {
		dense_all_gather(num_procs, num_active_procs, my_rank, vector_len,
						 reduced_chunk, chunk_boundaries, reduced_vector);
		return;
	}

	std::vector<MPI_Request> all_requests;
	all_requests.reserve(num_active_procs * 2); // Both send and recv

	// Send reduced_chunk to all other processors
	std::vector<int> send_data;
	if (my_rank < num_active_procs) {
		size_t my_chunk_size =
			chunk_boundaries[my_rank + 1] - chunk_boundaries[my_rank];

		if (reduced_chunk.size() * 2 < my_chunk_size) {
			// Sparse
			send_data.reserve(1 + 2 * reduced_chunk.size());
			send_data.push_back(0);

			int next_index = chunk_boundaries[my_rank];
			for (const auto& elem : reduced_chunk) {
				send_data.push_back(elem.first);
				send_data.push_back(elem.second);

				for (int i = next_index; i < elem.first; i++) {
					reduced_vector[i] = 0;
				}

				reduced_vector[elem.first] = elem.second;
				next_index = elem.first + 1;
			}

			for (int i = next_index; i < chunk_boundaries[my_rank + 1]; i++) {
				reduced_vector[i] = 0;
			}

		} else {
			// Dense
			send_data.reserve(1 + my_chunk_size);
			send_data.push_back(1);

			int next_index = chunk_boundaries[my_rank];
			for (const auto& elem : reduced_chunk) {
				for (int i = next_index; i < elem.first; i++) {
					send_data.push_back(0);
					reduced_vector[i] = 0;
				}

				send_data.push_back(elem.second);
				reduced_vector[elem.first] = elem.second;
				next_index = elem.first + 1;
			}

			for (int i = next_index; i < chunk_boundaries[my_rank + 1]; i++) {
				send_data.push_back(0);
				reduced_vector[i] = 0;
			}
		}

		for (int rank = 0; rank < num_procs; rank++) {
			if (rank == my_rank)
				continue;

			all_requests.emplace_back();
			MPI_Isend(send_data.data(), send_data.size(), MPI_INT, rank,
					  TAG_ALL_GATHER, MPI_COMM_WORLD, &all_requests.back());
		}
	}

	size_t num_send = all_requests.size();

	// Read reduced chunks from all active processors
	int* recv_buffer = new int[num_procs + 2 * vector_len];
	for (int rank = 0; rank < num_active_procs; rank++) {
		if (rank == my_rank) {
			continue;
		}

		size_t chunk_size = chunk_boundaries[rank + 1] - chunk_boundaries[rank];

		all_requests.emplace_back();
		MPI_Irecv(recv_buffer + chunk_boundaries[rank] * 2 + rank,
				  2 * chunk_size + 1, MPI_INT, rank, TAG_ALL_GATHER,
				  MPI_COMM_WORLD, &all_requests.back());
	}

	int index = -1;
	MPI_Status status;
	for (int i = 0; i < all_requests.size(); i++) {
		MPI_Waitany(all_requests.size(), all_requests.data(), &index, &status);

		if (index >= num_send) {
			int recv_size;
			MPI_Get_count(&status, MPI_INT, &recv_size);

			int rank = index - num_send;
			if (rank >= my_rank)
				rank += 1;

			int recv_index = chunk_boundaries[rank] * 2 + rank;
			if (recv_buffer[recv_index] == 0) {
				// Sparse
				int next_index = chunk_boundaries[rank];
				for (int i = 1; i < recv_size; i += 2) {
					int idx = recv_buffer[recv_index + i];
					int val = recv_buffer[recv_index + i + 1];

					for (int j = next_index; j < idx; j++) {
						reduced_vector[j] = 0;
					}

					reduced_vector[idx] = val;
					next_index = idx + 1;
				}

				for (int i = next_index; i < chunk_boundaries[rank + 1]; i++) {
					reduced_vector[i] = 0;
				}
			} else {
				// Dense
				for (int i = 1; i < recv_size; i++) {
					int idx = chunk_boundaries[rank] + i - 1;
					int val = recv_buffer[recv_index + i];
					reduced_vector[idx] = val;
				}
			}
		}
	}

	delete recv_buffer;
}

void rabenseifner_algorithm(const int num_procs, const int rank,
							const int vector_len,
							const std::map<int, int>& in_vector,
							const char* distribution, const double dist_param,
							int* reduced_vector) {
	std::vector<int> chunk_boundaries;
	if (!strcmp(distribution, "unknown")) {
		chunk_boundaries = estimate_partition_boundaries(num_procs, vector_len, in_vector);
	} else {
		chunk_boundaries = compute_partition_boundaries(num_procs, vector_len, distribution, dist_param);
	}
	int num_active_procs = chunk_boundaries.size() - 1;

	std::map<int, int> reduced_chunk =
		reduce_scatter(num_procs, num_active_procs, rank,
					   vector_len, in_vector, chunk_boundaries);

	//dense_all_gather(num_procs, num_active_procs, rank, vector_len,
	//				 reduced_chunk, chunk_boundaries, reduced_vector);
	//sparse_all_gather(num_procs, num_active_procs, rank, vector_len,
	//				  reduced_chunk, chunk_boundaries, reduced_vector);
	dynamic_all_gather(num_procs, num_active_procs, rank, vector_len,
					   reduced_chunk, chunk_boundaries, reduced_vector);
}


// ===================
// Recursive Doubling
// ===================

void recursive_double(const int num_procs, const int my_rank, const int vector_len,
					  const std::map<int, int>& in_vector, int *reduced_vector) {
	if (fmod(log2(num_procs), 1)) {
		if (my_rank == 0)
			std::cerr << "Error: recursive doubling requires the number of processors to be a power of two." << std::endl;
		return;
	}

	// Initialize reduced_vector with in_vector
	int next_index = 0;
	for (const auto& elem : in_vector) {
		for (int i = next_index; i < elem.first; i++) {
			reduced_vector[i] = 0;
		}
		reduced_vector[elem.first] = elem.second;
		next_index = elem.first + 1;
	}
	for (int i = next_index; i < vector_len; i++) {
		reduced_vector[i] = 0;
	}

	// Perform recursive doubling
    int* recv_data = new int[vector_len];
	MPI_Request* requests = new MPI_Request[2];
    for (int distance = 1; distance <= num_procs/2; distance *= 2) {
		// Send and receive requests
		int k = (my_rank + 1) % (distance * 2);
		int dest_rank = (k <= distance && k != 0)
			? my_rank + distance
			: my_rank - distance;

		MPI_Isend(reduced_vector, vector_len, MPI_INT, dest_rank, TAG_RECURSIVE_DOUBLE, MPI_COMM_WORLD, &requests[0]);
		MPI_Irecv(recv_data, vector_len, MPI_INT, dest_rank, TAG_RECURSIVE_DOUBLE, MPI_COMM_WORLD, &requests[1]);
		MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

		// Reduce receieved data
		for (int i = 0; i < vector_len; i++) {
			reduced_vector[i] += recv_data[i];
		}
    }
    delete recv_data;
    delete requests;
}


// =================
// Sparse Allreduce
// =================

void dist_sparse_all_reduce(const int num_procs, const int rank,
							const int vector_len,
							const std::map<int, int>& in_vector,
							const char* distribution, const double dist_param,
							int* reduced_vector) {
	if (false) { // TODO: determine threshold for when to use recursive doubling
		recursive_double(num_procs, rank, vector_len, in_vector, reduced_vector);
	} else {
		rabenseifner_algorithm(num_procs, rank, vector_len, in_vector, distribution, dist_param, reduced_vector);
	}
}
