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

int main()
{
return 0;
}
