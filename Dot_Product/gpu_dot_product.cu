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
	// Store this temporary sum of products (running sum) in the shared memory buffer
	buffer[buffer_idx] = temp_sum;

	// Point of synchronisation for all above threaded computations
	__syncthreads();

	// REDUCTIONS -- Summation Reduction --
	// Since we have 256 threads, to reduce this completely using two thread sum,
	// we will need x:2^x=256 iterations (x=8 iterations)
	int i = blockDim.x/2;
	while (i!=0)
	{
		if(buffer_idx < i)
		{
			// Add i'th element of each thread in buffer
			buffer[buffer_idx] += buffer[buffer_idx+i];
		}
		__syncthreads();
		// i then reduces by half since 256 elements would become 128 after 
		// the first reduction step
		i = i/2;
	}
	if(buffer_idx == 0)
	{
		// Add thread sum to block result, return control to CPU to perform
		// final arithmetic addition of block sums
		result[blockIdx.x] = cache[0];
	}
}
			
