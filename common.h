#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <map>

void naive_sparse_all_reduce(const int num_procs, const int rank,
							 const int vector_len,
							 const std::map<int, int>& in_vector,
							 std::map<int, int>& reduced_vector);

void dist_sparse_all_reduce(const int num_procs, const int rank,
							const int vector_len,
							const std::map<int, int>& in_vector,
							const char* distribution, const double dist_param,
							std::map<int, int>& reduced_vector);

#endif
