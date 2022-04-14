#include "common.h"
#include <math.h>
#include <vector>

#define TAG_REDUCE_SCATTER 0
#define TAG_ALL_GATHER 1
#define EPSILON 0.05 // TODO: determine optimal value for epsilon

// =================
// Helper Functions
// =================

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

/* Copies data to another vector, used at the start of direct allreduce to populate reduced vector with local data */
void copy_vector(const std::map<int, int>& in_vector, std::map<int, int>& reduced_vector) {


    for (auto i = in_vector.begin(); i != in_vector.end(); ++i) {
        reduced_vector[i->first] = i->second;
    }

}

/* Partitions {0,...,vector_len-1} into num_procs subsets of roughly equal mass
   according to the probability mass function specified by the distribution */
std::vector<int> compute_partition_chunks(const int num_procs, int &num_active_procs, const int my_rank, const int vector_len,
										  const char* distribution, const double dist_param) {
	auto pmf = [&](const int x, const char* distribution, const double dist_param) {
		if (!strcmp(distribution, "uniform")) {
			return 1/(double)dist_param;
		} else if (!strcmp(distribution, "geometric")) {
			return pow(1-dist_param, x-1) * dist_param;
		} else if (!strcmp(distribution, "poisson")) {
			return pow(dist_param, x) * exp(-dist_param) / factorial(x);
		}
		return (double)-1;
    };

	double total_mass = 0;
	for (int i = 0; i < vector_len; i++) {
		total_mass += pmf(i, distribution, dist_param);
	}

	int rank = 1;
	double mass = 0;
	const double uniform_mass = 1/(double)num_procs;
	std::vector<int> chunks; // chunks[i] is first index of processor i's chunk
	chunks.push_back(0);
	for (int idx = 0; idx < vector_len-1; idx++) {
		/* Assign indices to chunk until cumulative probability mass
		   is at least within EPSILON of (ideal) uniform mass */
		mass += pmf(idx, distribution, dist_param) / total_mass;
		if (mass >= uniform_mass - EPSILON) {
			chunks.push_back(idx+1);
			rank++; mass = 0;
		}
		if (rank == num_procs)
			break; // stop after last processor's chunk is determined
	}
	chunks.push_back(vector_len); // for convenience, so chunks[i+1] always exists
	num_active_procs = rank;

	// if (my_rank == 0) {
	// 	std::cout << "( ";
	// 	for (int i = 0; i < chunks.size(); i++) {
	// 		std::cout << chunks[i] << ' ';
	// 	}
	// 	std::cout << ")\n";
	// }

	return chunks;
}

// ===================
// Recursive Doubling
// ===================

/* Main recursive doubling operation */
void do_recursive_double(std::map<int, int>& vector, int *recv_data, 
                         const int rank, const int num_procs, const int n,
                         const int limit, const int distance) {
    
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

void reduce_scatter(const int num_procs, const int num_active_procs, const int my_rank, const int vector_len,
					const std::map<int, int>& in_vector, std::map<int, int>& reduced_chunk, std::vector<int> chunks) {
	// Scatter: For each active processor, send it the relevant chunk of in_vector
	for (int rank = 0; rank < num_active_procs; rank++) {
		if (rank == my_rank)
			continue;

		// Setup send data: [idx_1, val_1, ..., idx_nnz, val_nnz]
		std::vector<int> send_data;
		int send_size = 0;
		for (auto it = in_vector.upper_bound(chunks[rank]-1);
			 it != in_vector.lower_bound(chunks[rank+1]); it++) {
			send_data.push_back(it->first);
			send_data.push_back(it->second);
			send_size += 2;
		}

		MPI_Request send_request;
		MPI_Isend(send_data.data(), send_size, MPI_INT, rank, TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &send_request);
	}

	// Reduce: Receive and reduce all chunks from other processors
	if (my_rank < num_active_procs) {
		// Initialize reduced_chunk with in_vector
		for (auto it = in_vector.upper_bound(chunks[my_rank]-1);
			 it != in_vector.lower_bound(chunks[my_rank+1]); it++) {
			reduced_chunk[it->first] = it->second;
		}

		int* recv_data = new int[2*vector_len];
		for (int rank = 0; rank < num_procs; rank++) {
			if (rank == my_rank)
				continue;

			MPI_Status recv_status;
			int recv_size;
			MPI_Recv(recv_data, 2*vector_len, MPI_INT, rank, TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &recv_status);
			MPI_Get_count(&recv_status, MPI_INT, &recv_size);
			for (int i = 0; i < recv_size; i += 2) {
				int idx = recv_data[i];
				int val = recv_data[i+1];
				reduced_chunk[idx] += val;
			}
		}
		delete recv_data;
	}
}

void all_gather(const int num_procs, const int num_active_procs, const int my_rank, const int vector_len,
				const std::map<int, int>& reduced_chunk, std::map<int, int>& reduced_vector) {
	// Send reduced_chunk to all other processors
	if (my_rank < num_active_procs) {
		std::vector<int> send_data;
		for (auto i = reduced_chunk.begin(); i != reduced_chunk.end(); ++i) {
			reduced_vector[i->first] = i->second;
			send_data.push_back(i->first);
			send_data.push_back(i->second);
		}
		for (int dist = 1; dist < num_procs; dist++) {
			int dst = (my_rank+dist) % num_procs;
			MPI_Send(send_data.data(), 2*reduced_chunk.size(), MPI_INT, dst, TAG_ALL_GATHER, MPI_COMM_WORLD);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Read reduced chunks from all active processors
	int* recv_data = new int[2*vector_len];
	for (int rank = 0; rank < num_active_procs; rank++) {
		if (rank == my_rank)
			continue;

		MPI_Status recv_status;
		int recv_size;
		MPI_Recv(recv_data, 2*vector_len, MPI_INT, rank, TAG_ALL_GATHER, MPI_COMM_WORLD, &recv_status);
		MPI_Get_count(&recv_status, MPI_INT, &recv_size);
		for (int i = 0; i < recv_size; i += 2) {
			int idx = recv_data[i];
			int val = recv_data[i+1];
			reduced_vector[idx] = val;
		}
	}
	delete recv_data;
}

void sparse_all_reduce(const int num_procs, const int rank, const int vector_len,
					   std::map<int, int>& in_vector, std::map<int, int>& reduced_vector,
					   const char* distribution, const double dist_param) {
	int num_active_procs;
	std::vector<int> chunks = compute_partition_chunks(num_procs, num_active_procs, rank, vector_len, distribution, dist_param);
	std::map<int, int> reduced_chunk;
	reduce_scatter(num_procs, num_active_procs, rank, vector_len, in_vector, reduced_chunk, chunks);
	all_gather(num_procs, num_active_procs, rank, vector_len, reduced_chunk, reduced_vector);
	MPI_Barrier(MPI_COMM_WORLD);
}
