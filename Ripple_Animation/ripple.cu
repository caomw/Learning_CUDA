#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_bitmap.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_anim.h"

// Use NVCC compiler as shown to link OpenGL libraries:
// 	nvcc -o ripple.o ripple.cu -lGL -lglut

#define DIM 1024
#define PI 3.1415926535897932f


struct Data
{
	unsigned char *device_bitmap;
	CPUAnimBitmap *bitmap;
};

void garbagecollection(Data *d)
{
	cudaFree(d->device_bitmap);
}

__global__ void pixel_mapping( unsigned char *ptr, int ticks)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int incr = x + y * blockDim.x * gridDim.x;

	float px = x - DIM/2;
	float py = y - DIM/2;
	float euc_dist = sqrt((px*px)+(py*py));
	
	unsigned char clr = (unsigned char)( 128.0f + 127.0f * (cos(euc_dist/10.0f-ticks/7.0f)) / (euc_dist/10.0f + 1.0f) );

	ptr[incr*4 + 0] = clr;
	ptr[incr*4 + 1] = clr;
	ptr[incr*4 + 2] = clr;
	ptr[incr*4 + 3] = 255;
}

void generate_frame(Data *d, int ticks)
{
	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);
	pixel_mapping<<<blocks,threads>>>(d->device_bitmap,ticks);
	cudaMemcpy(d->bitmap->get_ptr(),d->device_bitmap,d->bitmap->image_size(),cudaMemcpyDeviceToHost);
}

int main(void)
{
	Data data;
	CPUAnimBitmap bitmap(DIM,DIM,&data);
	data.bitmap = &bitmap;
	cudaMalloc((void**)&data.device_bitmap,bitmap.image_size());
	bitmap.anim_and_exit((void (*)(void*,int))generate_frame,(void(*)(void*))garbagecollection);
}

