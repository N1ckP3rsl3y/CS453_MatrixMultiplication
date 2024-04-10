//Matrix multiplication that uses
//1-D A, B, C on both GPU and CPU (use FP32)


//CPU- reference implementation

//GPU- one output element per thread (MODE==1) //this is the original in matrix_multiply.cu
//GPU- one output element per thread with shared-memory tiling (MODE==4) //this is the original in matrix_multiply.cu

//Transposed matrix B:
//Same as MODE==1 above but matrix B is transposed (MODE==5)
//Same as MODE==4 above but matrix B is transposed (MODE==6)

//GPU- one element per thread but transpose matrix B so that we have more coalesced memory accesses (MODE==6)


#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>

#include <math.h>


// Constants for LxM * MxN
#define N 2048
#define M 4096
#define L 2048

#define MODE 1 //see implementations above for associated modes

// #define BLOCKDIMTILE 16


//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
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

// __global__ void matrixMultiOneElemPerThread(float *A, float *B, float *C, const unsigned int NUMELEM);
// __global__ void matrixMultiOneElemPerThreadTransposedMatrixB(float *A, float *B, float *C, const unsigned int NUMELEM);
// __global__ void matrixMultiOneElemPerThreadSharedMemoryTile(float *A, float *B, float *C, const unsigned int NUMELEM);
// __global__ void matrixMultiOneElemPerThreadSharedMemoryTileTransposedB(float *A, float *B, float *C, const unsigned int NUMELEM);

__global__ void matrixMultGeneralBaselineOneElemPerThread(float *A, float *B, float *C);

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

  	for (unsigned int i=0; i<M*N; i++){
  		B[i]=(float)rand()/(float)RAND_MAX;

  		//Transposed matrix
  		//Write code here to copy B into B_Transposed to populate the transposed matrix
        // B_Transposed[(i / N) + (i % N) * N] = B[i];
  	}

	printf("\nMemory requested for 5x NxN matrices (GiB) %f", (5.0*N*N*sizeof(float)/(1024.0*1024.0*1024.0)));

	///////////////////////////
	//CPU version:
	///////////////////////////

	printf("\nCommented sequential CPU execution");
	// computeMatrixCPU(A, B, C_CPU, N);

	//print matrix if N is <= 10x10
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

	//copy B to device (non-transposed)
	if (MODE==1 || MODE==4)
	{
		gpuErrchk(cudaMemcpy(dev_B, B, sizeof(float)*M*N, cudaMemcpyHostToDevice));
	}

	//copy B to device (transposed)
	else if (MODE==5 || MODE==6)
	{
		gpuErrchk(cudaMemcpy(dev_B, B_Transposed, sizeof(float)*N*M, cudaMemcpyHostToDevice));
	}

	//copy C to device (initialized to 0)
	gpuErrchk(cudaMemcpy(dev_C, C, sizeof(float)*L*N, cudaMemcpyHostToDevice));


	//execute kernel

	//MODE==1 refers to one thread per output element of the matrix
	if (MODE==1){
		printf("\nMODE==1");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		unsigned int BLOCKDIM = 1024;
		dim3 dimGrid(ceil(N*L*1.0/BLOCKDIM*1.0), 1, 1);
		dim3 dimBlock(BLOCKDIM, 1, 1);

		matrixMultGeneralBaselineOneElemPerThread<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);

	}
	else if (MODE==4){
		printf("\nMODE==4");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		//also for convenience, set the number of threads per block to be the shared memory tile size
		unsigned int BLOCKDIM = BLOCKDIMTILE; //blocks are of size BLOCKDIMTILE*BLOCKDIMTILE
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

		// matrixMultiOneElemPerThreadSharedMemoryTile<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	}
	//MODE==5 refers to one thread per output element of the matrix
	//uses transposed matrix B for coalesced global memory accesses
	else if (MODE==5){
		printf("\nMODE==5");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		unsigned int BLOCKDIM = 32; //blocks are of size 32x32=1024
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

		// matrixMultiOneElemPerThreadTransposedMatrixB<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

	}
	//MODE==6 refers to one thread per output element of the matrix
	//where shared-memory tiling is used, but works on the transposed matrix
	//similar to MODE==4
	//1-D grid (for the rows)
	else if (MODE==6){
		printf("\nMODE==6");
		//set number of blocks -- for convenience, use 2-D grid to represent matrix elements
		//also for convenience, set the number of threads per block to be the shared memory tile size
		unsigned int BLOCKDIM = BLOCKDIMTILE; //blocks are of size BLOCKDIMTILE*BLOCKDIMTILE
		dim3 dimGrid(ceil(N*1.0/BLOCKDIM*1.0), ceil(N*1.0/BLOCKDIM*1.0), 1);
		dim3 dimBlock(BLOCKDIM, BLOCKDIM, 1);

		// matrixMultiOneElemPerThreadSharedMemoryTileTransposedB<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

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
	outputSumElems(C, N*L);

	//print matrix if N is less than 16
	printMatrixIfSmall(C, N, 1);

	//Compare CPU and GPU matrices to determine if there are errors in floating point arithmetic or in the GPU code
	compareMatrices(C, C_CPU, N*N);


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
	for (int i=0; i<NUMELEM; i++)
	{
		sumElems += C[i];
	}
	printf("\nSum of elems in GPU output matrix (C): %f",sumElems);
}

void compareMatrices(float * C_GPU, float * C_CPU, unsigned int NUMELEM)
{
	float sumDelta=0;
	float maxDelta=0; //keep track of the maximum difference
	for (int i=0; i<L*M; i++)
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

	int ROW=0;
	int COL=0;

	for (ROW=0; ROW<L; ROW++)
		for (COL=0; COL<N; COL++)
			for (int k=0; k<M; k++)
			{
				C[(ROW*N)+COL]+=A[ROW*M+k]*B[COL+(k*N)];
			}

	double tendcpu=omp_get_wtime();
	printf("\nTime CPU: %f",tendcpu - tstartcpu);
}

void printMatrixIfSmall(float * C, const unsigned int NUMELEM, bool CPUGPU)
{
	int i, j;
	int cnt=0;
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

//each thread computes a single element of C using a row of A and column of B
__global__ void matrixMultGeneralBaselineOneElemPerThread(float *A, float *B, float *C)
{
    unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int ROW = tid / N;
    unsigned int COL = tid % N;

    if(tid < L * N)
    {
        // printf("%u %u %d %u\n", ROW, COL, L * N, tid);
        for(int iter = 0; iter < M; iter++)
        {
            C[ROW*N+COL] += A[ROW*M+iter] * B[iter*N+COL];
        }
    }

    return;
}

//matrix multiply
//each thread computes a single element of C using a row of A and column of B
// __global__ void matrixMultiOneElemPerThread(float *A, float *B, float *C, const unsigned int NUMELEM) {

//     unsigned int ROW = threadIdx.x+blockDim.x*blockIdx.x;
//     unsigned int COL = threadIdx.y+blockDim.y*blockIdx.y;

//     if(ROW < NUMELEM && COL < NUMELEM)
//     {
//         for (int k=0; k<NUMELEM; k++)
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

//     int COL = threadIdx.y+(blockDim.y*blockIdx.y);
//     int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
//     float localSum = 0.0;

//     __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
//     __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

//     for(int phase=0; phase<NUMELEM; phase+=BLOCKDIMTILE)
//     {
//         tileA[threadIdx.x][threadIdx.y]=A[ROW*NUMELEM+phase+threadIdx.y];

//         tileB[threadIdx.x][threadIdx.y]=B[(phase+threadIdx.x)*NUMELEM + COL];

//         __syncthreads();

//         for (int k=0; k<BLOCKDIMTILE; k++)
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
//         for (int k=0; k<NUMELEM; k++)
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

//     int COL = threadIdx.y+(blockDim.y*blockIdx.y);
//     int ROW = threadIdx.x+(blockDim.x*blockIdx.x);
//     float localSum = 0.0;

//     __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
//     __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

//     for(int phase=0; phase<NUMELEM; phase+=BLOCKDIMTILE)
//     {
//         tileA[threadIdx.x][threadIdx.y]=A[ROW*NUMELEM+phase+threadIdx.y];

//         tileB[threadIdx.x][threadIdx.y]=B[COL*NUMELEM+phase+threadIdx.x];

//         __syncthreads();

//         for (int k=0; k<BLOCKDIMTILE; k++)
//         {
//             localSum+=tileA[threadIdx.x][k]*tileB[k][threadIdx.y];
//         }

//         __syncthreads();
//     }

//     C[ROW*NUMELEM+COL] = localSum;

//     return;
// }
