#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"

// Block limit is 65,535 ; Thread limit is 512

__global__ void addvec_threads(int *vec1, int *vec2, int *sum)
{
	// Indexing scheme for multiple block and multiple thread setup
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		// Addition of vectors is defined as before
		sum[tid] = vec1[tid] + vec2[tid];
		// Thread increment is a function of the block and grid size
		// NOTE:
		// 	"We want each parallel thread to start on a different
		// data index, so we just need to take our thread and block
		// indexes and linearize them.."
		//
		// We can now work more elements than restricted by the hard
		// -ware limit of 65,535 * thread_count (<512)
		tid = tid + blockDim.x * gridDim.x;
	}
}


