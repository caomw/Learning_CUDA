#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_bitmap.h"

#define DIM 1000

struct complexNumber
{
	float real_part;
	float imaginary_part;
	
	// CONSTRUCTOR
	__device__ complexNumber(float x, float y) : real_part(x), imaginary_part(y) {}

	// MAGNITUDE OF A COMPLEX NUMBER
	__device__ float mag(void)
	{
	float mag_complexnum;
	mag_complexnum = (real_part*real_part)+(imaginary_part*imaginary_part);
	return mag_complexnum;
	}
	
	// MULTIPLICATION OF TWO COMPLEX NUMBERS
	__device__ complexNumber operator*(const complexNumber& x)
	{
	return complexNumber(real_part*x.real_part - imaginary_part*x.imaginary_part, imaginary_part*x.real_part + real_part*x.imaginary_part);
	}

	// ADDITION OF TWO COMPLEX NUMBERS
	__device__ complexNumber operator+(const complexNumber& y)
	{
	return complexNumber(real_part+y.real_part, imaginary_part+y.imaginary_part);
	}
};

__device__ int julia_set_verify(int a, int b)
{
float kx, ky;
// Convert from image co-ordinate system to imaginary plane - scaling and shifting
kx = ((float)(DIM/2 - a)/(DIM/2))*1.5;
ky = ((float)(DIM/2 - b)/(DIM/2))*1.5;

// Tweak this value. CUDA BY EXAMPLE suggested this value
complexNumber C(-0.8,0.156);
complexNumber Zn(kx,ky);

int iterator = 0;
for (iterator = 0; iterator < 200; iterator ++)
{
	Zn = (Zn*Zn) + C;
	if (Zn.mag()>1000) return 0;
}
return 1;
}

int main()
{
return 0;
}
