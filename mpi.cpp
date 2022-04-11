#include "common.h"
#include <mpi.h>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>

// =================
// Helper Functions
// =================

/* Modulus operator without a negative output */
int mod(int a, int b) {
	int res = a%b;
	return res < 0 ? res+b : res;
}

/* Populates a vector with the contents of a map */
void map_to_vector(const std::map<int, int>& m, std::vector<int>& v) {
	for (auto i = m.begin(); i != m.end(); ++i) {
		v[i->first] = i->second;
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

/* Sums local_data and recv_data, writing the result to local_data */
void operate(std::map<int, int>& local_data, std::map<int, int>& recv_data) {
	for (auto i = recv_data.begin(); i != recv_data.end(); ++i) {
		local_data[i->first] += i->second;
	}
}

/* Halves the input map */
void halve(std::map<int, int> m, bool odd) {
	int limit = m.size()/2;
	int cnt = 0;
	if (odd) { //first half
		for (std::map<int, int>::iterator i = m.begin(); cnt < limit; ++i) {
			m.erase(i);
			cnt++;
		}
	} else { //second half
		for (std::map<int, int>::iterator i = m.begin(); i != m.end(); ++i) {
			if (cnt>=limit)
				m.erase(i);
		}
	}
}

// =================
// Direct Allreduce
// =================

/* Main operation */
void reduce_layer(std::map<int, int>& data, const std::map<int, int>& in_vector,
				  const int dist, const int rank, const int num_procs, const int n) {
	MPI_Request *requests = new MPI_Request[2];
	MPI_Request r1, r2;
	requests[0] = r1;
	requests[1] = r2;

	//Setup send data
	std::vector<int> send_data;
	send_data.resize(n);
	map_to_vector(in_vector, send_data);

	//Send and receive data
	int dest;
	dest = (rank+dist)%num_procs;
	MPI_Isend(send_data.data(), n, MPI_INT, dest, 0, MPI_COMM_WORLD, &requests[0]);
	dest = mod(rank-dist, num_procs);

	int *recv_data = new int[n];
	MPI_Irecv(recv_data, n, MPI_INT, dest, 0, MPI_COMM_WORLD, &requests[1]);
	MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

	//Operate on received data
	std::map<int, int> recv_map;
	array_to_map(recv_data, n, recv_map);
	operate(data, recv_map);

	delete recv_data;
}

void sparse_all_reduce(const int num_procs, const int rank,
					   const int vector_len,
					   const std::map<int, int>& in_vector,
					   std::map<int, int>& reduced_vector) {
	copy_vector(in_vector, reduced_vector);
	for (int dist = 1; dist < num_procs; ++dist) {
		reduce_layer(reduced_vector, in_vector, dist, rank, num_procs, vector_len);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
