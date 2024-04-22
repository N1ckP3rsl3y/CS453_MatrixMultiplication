//Matrix multiplication that uses
//1-D A, B, C on both GPU and CPU (use FP32)


#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>

#include <math.h>


// Constants for LxM * MxN
#define L 16384
#define M 16384
#define N 16384

#define MODE 4 //see implementations above for associated modes

#define BLOCKDIMTILE 8


//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, unsigned int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;


//function prototypes
void warmUpGPU();
void compareMatrices(float * C_GPU, float * C_CPU, unsigned int NUMELEM);
void printMatrixIfSmall(float * C, const unsigned int NUMELEM, bool CPUGPU);
void computeMatrixCPU(float * A, float * B, float *C, const unsigned int NUMELEM);
void outputSumElems(float * C, unsigned int NUMELEM);

__global__ void matrixMultGeneralBaselineOneElemPerThread(float *A, float *B, float *C);
__global__ void matrixMultGeneralBaselineOneElemPerThread2D(float *A, float *B, float *C);
__global__ void matrixMultGeneralOneElemPerThread2DTile(float *A, float *B, float *C);
__global__ void matrixMultGeneralOneElemPerThread2DTileTransposed(float *A, float *B, float *C);
__global__ void matrixMultGeneralOneElemPerThread2DNoBankConflict1(float *A, float *B, float *C);

int main(int argc, char *argv[])
{


	warmUpGPU();

	//change OpenMP settings
	//disregard --- only for parallel CPU version if applicable
	omp_set_num_threads(1);

  	//seed random number generator with constant seed
  	srand(123);

  	float * A;
  	float * B;
  	float * B_Transposed;
  	float * C;
  	float * C_CPU;

  	A=(float *)malloc(sizeof(float)*L*M);
  	B=(float *)malloc(sizeof(float)*M*N);
  	B_Transposed=(float *)malloc(sizeof(float)*N*M);
  	C=(float *)calloc(L*N,sizeof(float));
  	C_CPU=(float *)calloc(L*N,sizeof(float));


  	//init matrices of FP32 with random numbers between 0 and 1
    for(unsigned int i = 0; i < L*M; i++) {
  		A[i]=(float)rand()/(float)RAND_MAX;
    }

  	for(unsigned int i=0; i<M*N; i++){
  		B[i]=(float)rand()/(float)RAND_MAX;

  		//Transposed matrix
  		//Write code here to copy B into B_Transposed to populate the transposed matrix
        B_Transposed[(i / N) + (i % N) * M] = B[i];
  	}

	printf("\nMemory requested for 5x NxN matrices (GiB) %f", (5.0*L*N*sizeof(float)/(1024.0*1024.0*1024.0)));

	///////////////////////////
	//CPU version:
	///////////////////////////

	printf("\nCommented sequential CPU execution");
	// computeMatrixCPU(A, B, C_CPU, N);

	printMatrixIfSmall(C_CPU, N, 0);


	/////////////////////////////
	//GPU
	////////////////////////////


	double tstart=omp_get_wtime();


	cudaError_t errCode=cudaSuccess;

	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl;
	}

	float * dev_A;
	float * dev_B;
	float * dev_C;

	unsigned int * debug;
	debug=(unsigned int *)malloc(sizeof(unsigned int));
	*debug=0;

	//allocate on the device: A, B, C
	gpuErrchk(cudaMalloc((float**)&dev_A, sizeof(float)*L*M));
	gpuErrchk(cudaMalloc((float**)&dev_B, sizeof(float)*M*N));
	gpuErrchk(cudaMalloc((float**)&dev_C, sizeof(float)*L*N));

	//copy A to device
	gpuErrchk(cudaMemcpy(dev_A, A, sizeof(float)*L*M, cudaMemcpyHostToDevice));

	//copy B to device (transposed)
	if (MODE==4)
	{
		gpuErrchk(cudaMemcpy(dev_B, B_Transposed, sizeof(float)*N*M, cudaMemcpyHostToDevice));
	}
    else // Otherwise, copy B (non-transposed) to device
    {
		gpuErrchk(cudaMemcpy(dev_B, B, sizeof(float)*M*N, cudaMemcpyHostToDevice));
    }

	//copy C to device (initialized to 0)
	gpuErrchk(cudaMemcpy(dev_C, C, sizeof(float)*L*N, cudaMemcpyHostToDevice));


	//execute kernel

	if (MODE==1){ // 1D blocks/grids
		printf("\nMODE == 1");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		unsigned int BLOCKDIM = 1024;

		matrixMultGeneralBaselineOneElemPerThread<<<ceil(N*L*1.0/BLOCKDIM*1.0), BLOCKDIM>>>(dev_A, dev_B, dev_C);
	}
    else if(MODE == 2) { // 2D blocks/grids
		printf("\nMODE == 2");
        unsigned int BLOCKDIM = BLOCKDIMTILE;
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(L*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

        matrixMultGeneralBaselineOneElemPerThread2D<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    } else if(MODE == 3) { // 2D tiling
        printf("\nMODE == 3");
        unsigned int BLOCKDIM = BLOCKDIMTILE;
		dim3 dimGrid(ceil(L*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

        matrixMultGeneralOneElemPerThread2DTile<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    } else if(MODE == 4) { // 2D tiling; transposed B
        printf("\nMODE == 4");
        unsigned int BLOCKDIM = BLOCKDIMTILE;
		dim3 dimGrid(ceil(L*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

        matrixMultGeneralOneElemPerThread2DTileTransposed<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    } else if(MODE == 5) {
        printf("\nMODE == 5");
        unsigned int BLOCKDIM = BLOCKDIMTILE;
		dim3 dimGrid(ceil(L*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

        matrixMultGeneralOneElemPerThread2DNoBankConflict1<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    }
	else
	{
		printf("Error: incorrect mode\n");
		return 0;
	}


	//check kernel errors
	errCode=cudaGetLastError();
	if(errCode != cudaSuccess) {
	cout << "\nError: GPU kernel had an error with code: " << errCode << endl;
	}

	//end execute kernel

	//Copy C from the GPU
	gpuErrchk(cudaMemcpy(C, dev_C, sizeof(float)*L*N, cudaMemcpyDeviceToHost));

	double tend=omp_get_wtime();

	printf("\nTotal time GPU (s): %f",tend-tstart);


	//print sum of elements in GPU array C
	outputSumElems(C, L*N);

	//print matrix if N is less than 16
	printMatrixIfSmall(C, L*N, 1);

	//Compare CPU and GPU matrices to determine if there are errors in floating point arithmetic or in the GPU code
	compareMatrices(C, C_CPU, L*N);


	printf("\n");

	//free memory
	free(A);
  	free(B);
  	free(B_Transposed);
  	free(C);
  	free(C_CPU);
  	cudaFree(dev_A);
  	cudaFree(dev_B);
  	cudaFree(dev_C);

	return 0;
}


void outputSumElems(float * C, unsigned int NUMELEM)
{
	float sumElems=0;
	for (unsigned int i=0; i<NUMELEM; i++)
	{
		sumElems += C[i];
	}
	printf("\nSum of elems in GPU output matrix (C): %f",sumElems);
}

void compareMatrices(float * C_GPU, float * C_CPU, unsigned int NUMELEM)
{
	float sumDelta=0;
	float maxDelta=0; //keep track of the maximum difference
	for (unsigned int i=0; i<L*N; i++)
	{
		float delta = fabs(C_CPU[i]-C_GPU[i]);
		sumDelta += delta;
		if(maxDelta<delta)
		{
			maxDelta = delta;
		}
	}

	printf("\nSum of deltas between matrices: %f",sumDelta);
	printf("\nMaximum delta between elements between matrices: %f",maxDelta);
}




void warmUpGPU(){
printf("\nWarming up GPU for time trialing...\n");
cudaDeviceSynchronize();
return;
}


void computeMatrixCPU(float * A, float * B, float * C, const unsigned int NUMELEM)
{
	double tstartcpu=omp_get_wtime();

	unsigned int ROW=0;
	unsigned int COL=0;

	for (ROW=0; ROW<L; ROW++)
		for (COL=0; COL<N; COL++)
			for (unsigned int k=0; k<M; k++)
			{
				C[(ROW*N)+COL]+=A[ROW*M+k]*B[COL+(k*N)];
			}

	double tendcpu=omp_get_wtime();
	printf("\nTime CPU: %f",tendcpu - tstartcpu);
}

void printMatrixIfSmall(float * C, const unsigned int NUMELEM, bool CPUGPU)
{
	unsigned int i, j;
	unsigned int cnt=0;
	if (N<=16)
	{
		if(CPUGPU==0)
			printf("\nCPU matrix is: \n");
		else
			printf("\nGPU matrix is: \n");

		for (i=0; i<L; i++){
			for (j=0; j<N; j++){
				printf("%.2f, ",C[cnt]);
				cnt++;
			}
			printf("\n");
		}
	}
}

// MODE 1
__global__ void matrixMultGeneralBaselineOneElemPerThread(float *A, float *B, float *C)
{
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int ROW = tid / N;
    unsigned int COL = tid % N;

    if(tid < L * N)
    {
        for(unsigned int iter = 0; iter < M; iter++)
        {
            C[ROW*N+COL] += A[ROW*M+iter] * B[iter*N+COL];
        }
    }

    return;
}

// MODE 2
__global__ void matrixMultGeneralBaselineOneElemPerThread2D(float *A, float *B, float *C)
{
    unsigned int ROW = threadIdx.y+blockDim.y*blockIdx.y;
    unsigned int COL = threadIdx.x+blockDim.x*blockIdx.x;

    if(ROW < L && COL < N)
    {
        for (unsigned int k=0; k<M; k++)
        {
            C[ROW*N + COL]+=A[ROW*M + k]*B[k*N + COL];
        }
    }

    return;
}

// MODE 3
__global__ void matrixMultGeneralOneElemPerThread2DTile(float *A, float *B, float *C)
{
    unsigned int COL = threadIdx.y+(blockDim.y*blockIdx.y);
    unsigned int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
    float localSum = 0.0;

    __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
    __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

    for(unsigned int phase=0; phase<M; phase+=BLOCKDIMTILE)
    {
        if((ROW*M+phase+threadIdx.y) < L * M) {
            tileA[threadIdx.x][threadIdx.y] = A[ROW*M+phase+threadIdx.y];
        } else {
            tileA[threadIdx.x][threadIdx.y] = 0.0;
        }

        if(((phase+threadIdx.x)*N + COL) < M * N) {
            tileB[threadIdx.x][threadIdx.y] = B[(phase+threadIdx.x)*N + COL];
        } else {
            tileB[threadIdx.x][threadIdx.y] = 0.0;
        }

        __syncthreads();

        for (unsigned int k=0; k<BLOCKDIMTILE; k++) {
            localSum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
        }

        __syncthreads();
    }

    C[ROW*N+COL] = localSum;

    return;
}

// MODE 4
__global__ void matrixMultGeneralOneElemPerThread2DTileTransposed(float *A, float *B, float *C)
{
    unsigned int COL = threadIdx.y+(blockDim.y*blockIdx.y);
    unsigned int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
    float localSum = 0.0;

    __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
    __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

    for(unsigned int phase=0; phase<M; phase+=BLOCKDIMTILE)
    {
        if((ROW*M+phase+threadIdx.y) < L * M) {
            tileA[threadIdx.x][threadIdx.y] = A[(ROW*M+phase+threadIdx.y)];
        } else {
            tileA[threadIdx.x][threadIdx.y] = 0.0;
        }

        if((COL*M+phase+threadIdx.x) < M * N) {
            tileB[threadIdx.x][threadIdx.y] = B[COL*M+phase+threadIdx.x];
        } else {
            tileB[threadIdx.x][threadIdx.y] = 0.0;
        }

        __syncthreads();

        for (unsigned int k=0; k<BLOCKDIMTILE; k++)
        {
            localSum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
        }

        __syncthreads();
    }

    C[ROW*N+COL] = localSum;

    return;
}

// MODE 5
__global__ void matrixMultGeneralOneElemPerThread2DNoBankConflict1(float *A, float *B, float *C)
{
    unsigned int COL = threadIdx.y+(blockDim.y*blockIdx.y);
    unsigned int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
    float localSum = 0.0;

    __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
    __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

    for(unsigned int phase=0; phase<M; phase+=BLOCKDIMTILE)
    {
        if((ROW*M+phase+threadIdx.y) < L * M) {
            tileA[threadIdx.x][threadIdx.y] = A[ROW*M+phase+threadIdx.y];
        } else {
            tileA[threadIdx.x][threadIdx.y] = 0.0;
        }

        if(((phase+threadIdx.x)*N + COL) < M * N) {
            tileB[threadIdx.x][threadIdx.y] = B[(phase+threadIdx.x)*N + COL];
        } else {
            tileB[threadIdx.x][threadIdx.y] = 0.0;
        }

        __syncthreads();

        for (unsigned int k=0; k<BLOCKDIMTILE; k++) {
            localSum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
        }

        __syncthreads();
    }

    C[ROW*N+COL] = localSum;

    return;
}


//matrix multiply
//each thread computes a single element of C using a row of A and column of B
// __global__ void matrixMultiOneElemPerThread(float *A, float *B, float *C, const unsigned int NUMELEM) {

//     unsigned int ROW = threadIdx.x+blockDim.x*blockIdx.x;
//     unsigned int COL = threadIdx.y+blockDim.y*blockIdx.y;

//     if(ROW < NUMELEM && COL < NUMELEM)
//     {
//         for (unsigned int k=0; k<NUMELEM; k++)
//         {
//             C[ROW*NUMELEM + COL]+=A[ROW*NUMELEM + k]*B[k*NUMELEM + COL];
//         }
//     }

//     return;
// }

//matrix multiply
//each thread computes a single element of C using a row of A and column of B
//uses shared memory to tile the computation to eliminate extra accesses to global memory
//This example is from Chapter 5 in the textbook with some minor modifications
// __global__ void matrixMultiOneElemPerThreadSharedMemoryTile(float *A, float *B, float *C, const unsigned int NUMELEM) {

//     unsigned int COL = threadIdx.y+(blockDim.y*blockIdx.y);
//     unsigned int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
//     float localSum = 0.0;

//     __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
//     __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

//     for(unsigned int phase=0; phase<NUMELEM; phase+=BLOCKDIMTILE)
//     {
//         tileA[threadIdx.x][threadIdx.y]=A[ROW*NUMELEM+phase+threadIdx.y];

//         tileB[threadIdx.x][threadIdx.y]=B[(phase+threadIdx.x)*NUMELEM + COL];

//         __syncthreads();

//         for (unsigned int k=0; k<BLOCKDIMTILE; k++)
//         {
//             localSum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
//         }

//         __syncthreads();
//     }

//     C[ROW*NUMELEM+COL] = localSum;

//     return;
// }

//matrix multiply
//each thread computes a single element of C using a row of A and a row of B
//Matrix B is transposed to allow coalesced memory accesses
// __global__ void matrixMultiOneElemPerThreadTransposedMatrixB(float *A, float *B, float *C, const unsigned int NUMELEM) {

//     unsigned int COL = threadIdx.x+blockDim.x*blockIdx.x;
//     unsigned int ROW = threadIdx.y+blockDim.y*blockIdx.y;

//     if(ROW < NUMELEM && COL < NUMELEM)
//     {
//         for (unsigned int k=0; k<NUMELEM; k++)
//         {
//             C[ROW*NUMELEM+COL]+=A[ROW*NUMELEM+k]*B[COL*NUMELEM+k];
//         }
//     }

//     return;
// }

//matrix multiply
//each thread computes a single element of C using a row of A and column of B
//uses shared memory to tile the computation to eliminate extra accesses to global memory
//This example is from Chapter 5 in the textbook with some minor modifications
// __global__ void matrixMultiOneElemPerThreadSharedMemoryTileTransposedB(float *A, float *B, float *C, const unsigned int NUMELEM) {

//     unsigned int COL = threadIdx.y+(blockDim.y*blockIdx.y);
//     unsigned int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
//     float localSum = 0.0;

//     __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
//     __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

//     for(unsigned int phase=0; phase<NUMELEM; phase+=BLOCKDIMTILE)
//     {
//         tileA[threadIdx.x][threadIdx.y]=A[ROW*NUMELEM+phase+threadIdx.y];

//         tileB[threadIdx.x][threadIdx.y]=B[COL*NUMELEM+phase+threadIdx.x];

//         __syncthreads();

//         for (unsigned int k=0; k<BLOCKDIMTILE; k++)
//         {
//             localSum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
//         }

//         __syncthreads();
//     }

//     C[ROW*NUMELEM+COL] = localSum;

//     return;
// }
