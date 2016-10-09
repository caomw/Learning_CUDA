#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"

const int threads_per_block = 256;
const int N = 33 * 1024;

// Define a macro for finding the smaller of two integers
#define min_value(a,b) (a<b?a:b)

__global__ dotprod(float *vector1, float *vector2, float *result)
{
	// Create a buffer on the GPU shared memory to store the intermediate sums
	__shared__ float buffer[threads_per_block];
	// Find thread index using block offset
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	// The buffer index will be the same as the thread index since each thread 
	// will calculate it's own sum and that sum will be stored in the shared 
	// memory
	int buffer_idx = threadIdx.x;
	float temp_sum = 0;
	while (thread_id < N)
	{
		// Calculate the intermediate products and store in a temp variable
		temp_sum = temp_sum + vector1[thread_id]*vector2[thread_id];
		// Increment thread element index by total number of elements in thread
		thread_id = thread_id + blockDim.x * gridDim.x;
	}
	// Store this temporary sum of products in the shared memory buffer
	buffer[buffer_idx] = temp_sum;

