#include "common.h"
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <boost/math/distributions/geometric.hpp>
#include <boost/math/distributions/poisson.hpp>

#define TAG_REDUCE_SCATTER 0
#define TAG_ALL_GATHER 1
#define EPSILON 0.00001

// =================
// Helper Functions
// =================

std::vector<int> estimate_partition_boundaries(const int num_procs, const int vector_len,
                                               const std::map<int, int>& in_vector) {
    int num_active_procs = num_procs;
    if (num_active_procs > in_vector.size())
        num_active_procs = in_vector.size();

    std::vector<int64_t> tmp_boundaries;
    tmp_boundaries.reserve(num_active_procs + 1);
    tmp_boundaries.emplace_back(0);

    int chunk_size = in_vector.size() / num_active_procs;

    int i = 0;
    for (const auto& elem : in_vector) {
        if (i >= chunk_size) {
            tmp_boundaries.emplace_back(elem.first);
            i = 0;

            if (in_vector.size() % num_active_procs > 0
                    && tmp_boundaries.size() == in_vector.size() % num_active_procs) {
                chunk_size += 1;
            }
        }

        i++;
    }

    if (tmp_boundaries.back() < vector_len)
        tmp_boundaries.emplace_back(vector_len);

    int64_t* recv_buf = new int64_t[tmp_boundaries.size()];
    MPI_Allreduce(tmp_boundaries.data(), recv_buf, tmp_boundaries.size(),
                  MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    std::vector<int> boundaries;
    boundaries.reserve(tmp_boundaries.size());
    for (int i = 0; i < tmp_boundaries.size(); i++) {
        double avg = double(recv_buf[i]) / num_procs;
        boundaries.emplace_back(avg + 0.5);
    }

    return std::move(boundaries);
}

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

/* Partitions {0,...,vector_len-1} into num_procs subsets of roughly equal mass
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

/* Modulus operator without a negative output */
long factorial(const int n) {
    long f = 1;
    for (int i = 2; i <= n; i++)
        f *= i;
    return f;
}

/* Populates a vector with the contents of a map */
void map_to_vector(const std::map<int, int>& m, std::vector<int>& v) {

    for (auto i = m.begin(); i != m.end(); ++i) {
        v[i->first] = i->second; 
    }

}

/* Populates a map with the contents of a vector */
void vector_to_map(std::map<int, int>& m, std::vector<int>& v) {
    for (int i=0; i<v.size(); ++i) {
        m[i] = v[i];
    }
}

/* Populates a map with the contents of an array */
void array_to_map(int *v, int n, std::map<int, int>& m) {
    
    for (int i = 0; i < n; ++i) {
        m[i] = v[i];
    }

}

// ===================
// Recursive Doubling
// ===================

/* Main recursive doubling operation */
void do_recursive_double(std::map<int, int>& vector, int *recv_data, 
                         const int rank, const int num_procs, const int n,
                         const int limit, const int distance) {
	//Error if num_procs is not a power of two or if it's less than 1
	if (fmod(log2(num_procs), 1) || num_procs < 1)
		std::cout << "Error: recursive doubling should only be done when the number of processors is a power of two." << std::endl;

    MPI_Request *requests = new MPI_Request[2];
    MPI_Request r1;
    MPI_Request r2;
    requests[0] = r1;
    requests[1] = r2;

    //Populate send_data
    std::vector<int> send_data;
    send_data.resize(n*2);
    int idx = 0;

    for (auto i = vector.begin(); i != vector.end(); ++i) {
        send_data[idx] = i->first; //index
        send_data[idx+1] = i->second; //value
        idx+=2;
    }

    //Send and receive requests
    if ( ((rank + 1) % (distance * 2)) <= distance && ((rank + 1) % (distance * 2)) ) { 
        ///std::cout << "Rank " << rank << " sending to process " << rank+distance << std::endl;
        MPI_Isend(send_data.data(), n*2, MPI_INT, rank+distance, NULL, MPI_COMM_WORLD, &requests[0]);
        //std::cout << "Rank " << rank << " sending to process " << rank-distance << std::endl;
        //std::cout << "Rank " << rank << " receiving from process " << rank+distance << std::endl;
        MPI_Irecv(recv_data, n*2, MPI_INT, rank+distance, NULL, MPI_COMM_WORLD, &requests[1]);
        //std::cout << "Rank " << rank << " receiving from process " << rank-distance << std::endl;
    } else {
        //std::cout << "Rank " << rank << " sending to process " << rank-distance << std::endl;
        MPI_Isend(send_data.data(), n*2, MPI_INT, rank-distance, NULL, MPI_COMM_WORLD, &requests[0]);
        //std::cout << "Rank " << rank << " sending to process " << rank+distance << std::endl;
        //std::cout << "Rank " << rank << " receiving from process " << rank-distance << std::endl;
        MPI_Irecv(recv_data, n*2, MPI_INT, rank-distance, NULL, MPI_COMM_WORLD, &requests[1]);
        //std::cout << "Rank " << rank << " receiving from process " << rank+distance << std::endl;
    }
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    
    //operate on receieved data
    for (int i = 0; i < n*2; i+=2) {
        vector[recv_data[i]] += recv_data[i+1];
    }
    
    delete requests;
}


/* Recursive doubling loop */
void recursive_double(std::map<int, int>& in_vector, 
                      std::map<int, int>& reduced_vector, const int num_procs, 
                      const int rank, const int vector_len) {

    int distance = 1;
    int limit = num_procs/2;
    int *recv_data = new int[vector_len*2]; //*2 because it needs to fit the indices as well

    //Main loop
    while (distance <= limit) {
        do_recursive_double(in_vector, recv_data, rank, num_procs, vector_len, limit, distance);
        distance *= 2;
    }
    
    //Copy to reduced vector
    for (auto i = in_vector.begin(); i != in_vector.end(); ++i) {
        reduced_vector[i->first] = i->second;
    }

    delete recv_data;
}


// =================
// Direct Allreduce
// =================

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

void all_gather(const int num_procs, const int num_active_procs,
                const int my_rank, const int vector_len,
                const std::map<int, int>& reduced_chunk,
                const std::vector<int> chunk_boundaries,
                std::map<int, int>& reduced_vector) {
    std::vector<MPI_Request> all_requests;
    all_requests.reserve(num_active_procs * 2); // Both send and recv

    for (const auto& elem : reduced_chunk) {
        reduced_vector.emplace(elem.first, elem.second);
    }

    // Send reduced_chunk to all other processors
    std::vector<int> send_data; 
    send_data.reserve(2 * reduced_chunk.size());

    if (my_rank < num_active_procs) {
        for (const auto& elem : reduced_chunk) {
            send_data.push_back(elem.first);
            send_data.push_back(elem.second);
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

    std::vector<int*> next_buffer;
    next_buffer.emplace_back(recv_buffer);

    for (int rank = 0; rank < num_active_procs; rank++) {
        if (rank == my_rank)
            continue;

        size_t chunk_size = chunk_boundaries[rank + 1] - chunk_boundaries[rank];

        all_requests.emplace_back();
        MPI_Irecv(next_buffer.back(), 2 * chunk_size, MPI_INT, rank,
                  TAG_ALL_GATHER, MPI_COMM_WORLD, &all_requests.back());

        next_buffer.emplace_back(next_buffer.back() + 2 * chunk_size);
    }

    int index = -1;
    MPI_Status status;
    for (int i = 0; i < all_requests.size(); i++) {
        MPI_Waitany(all_requests.size(), all_requests.data(), &index, &status);

        if (index >= num_send) {
            int recv_size;
            MPI_Get_count(&status, MPI_INT, &recv_size);

            for (int i = 0; i < recv_size; i += 2) {
                reduced_vector.emplace(next_buffer[index - num_send][i],
                                       next_buffer[index - num_send][i+1]);
            }
        }
    }

    delete recv_buffer;
}

void dist_sparse_all_reduce(const int num_procs, const int rank,
                            const int vector_len,
                            const std::map<int, int>& in_vector, 
                            const char* distribution, const double dist_param,
                            std::map<int, int>& reduced_vector) {
    // std::vector<int> chunk_boundaries =
    //  estimate_partition_boundaries(num_procs, vector_len, in_vector);
    std::vector<int> chunk_boundaries =
        compute_partition_boundaries(num_procs, vector_len, distribution, dist_param);
    int num_active_procs = chunk_boundaries.size() - 1;

    if (rank == 0) {
        std::cout << '[' << num_active_procs << "]";
        for (const auto x : chunk_boundaries) {
            std::cout << ' ' << x;
        }
        std::cout << '\n';
    }

    std::map<int, int> reduced_chunk =
        reduce_scatter(num_procs, num_active_procs, rank,
                       vector_len, in_vector, chunk_boundaries);

    all_gather(num_procs, num_active_procs, rank, vector_len,
               reduced_chunk, chunk_boundaries, reduced_vector);
}