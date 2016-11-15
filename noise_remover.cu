#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MATCH(s) (!strcmp(argv[ac], (s)))
#define STAT_BLOCK_SIZE 512
#define COMPUTE_THREAD_SIZE

// returns the current time
static const double kMicro = 1.0e-6;
double get_time() {
  struct timeval TV;
  struct timezone TZ;
  const int RC = gettimeofday(&TV, &TZ);
  if(RC == -1) {
    printf("ERROR: Bad call to gettimeofday\n");
    return(-1);
  }
  return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}

__global__ void sumArray(double *out, unsigned char *in, int size) {
  //if (blockIdx.x <= blockNum ) {
  extern __shared__ double sdata[];

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  if (index < size)
    sdata[tid] = (double) in[index];
  else
    sdata[tid] = 0;
  __syncthreads(); // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }
  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }

}

__global__ void sum2Array(double *out, unsigned char *in, int size) {
  //if (blockIdx.x <= blockNum ) {
  extern __shared__ double sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  if (myId < size)
    sdata[tid] = (double)in[myId] * (double)in[myId];
  else
    sdata[tid] = 0;
  __syncthreads(); // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }
  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}
//__global__ void naiveSum(double *dsums,

/*Parallel computation of compute 1*/
__global__ void compute1(double *north_deriv, double *south_deriv, double *west_deriv,
			 double *east_deriv, double *diff_coef,
			 unsigned char *image, int width, int height, double std_dev)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= 1 && j >= 1 && j < width-1 && i < height-1) {
    int k = i * width + j;
    unsigned char currentPixel = image[k]; //Register for current pixel
    double ndk, sdk, wdk, edk; //Registers for derivatives
    double coef; //Register for diff coefficient
    double laplacian, num, den, std_dev2, gradient_square;
    ndk = image[(i - 1) * width + j] - currentPixel;	// north derivative --- 1 doubleing point arithmetic operations
    sdk = image[(i + 1) * width + j] - currentPixel;	// south derivative --- 1 doubleing point arithmetic operations
    wdk = image[i * width + (j - 1)] - currentPixel; // west derivative --- 1 doubleing point arithmetic operations
    edk = image[i * width + (j + 1)] - currentPixel;	   // east derivative --- 1 doubleing point arithmetic operations
    gradient_square = (ndk * ndk + sdk * sdk + wdk * wdk + edk * edk) / (currentPixel * currentPixel);
    laplacian = (ndk + sdk + wdk + edk) / currentPixel; // 4 doubleing point arithmetic operations
    num = (0.5 * gradient_square) - ((1.0 / 16.0) * (laplacian * laplacian)); // 5 doubleing point arithmetic operations
    den = 1 + (.25 * laplacian); // 2 doubleing point arithmetic operations
    std_dev2 = num / (den * den); // 2 doubleing point arithmetic operations
    den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 doubleing point arithmetic operations
    coef = 1.0 / (1.0 + den); // 2 doubleing point arithmetic operations -
    if (coef < 0) {
      diff_coef[k] = 0;
    } else if (coef > 1) {
      diff_coef[k] = 1;
    }
    //Save derivatives in registers to device memory for later use(in compute 2)
    north_deriv[k] = ndk;
    south_deriv[k] = sdk;
    west_deriv[k] = wdk;
    east_deriv[k] = edk;
  }
}

/*if (localj == 16) local_image[locali][localj+1] = image[k+1];
  else if (localj == 1) local_image[locali][localj-1] = image[k-1];
  else if (locali == 16) local_image[locali+1][localj] = image[k+width];*/
//else if (locali == 1) local_image[locali-1][localj] = image[k-width];
//else if (locali == 1) local_image[locali-1][localj] = image[k-width];
/*if (localj == 16) {//COMPUTE_THREAD_SIZE) {
  local_image[locali][localj+1] = image[k+1];
  }
  else if (localj == 1) {
  local_image[locali][localj-1] = image[k-1];
  }
  else if (locali == COMPUTE_THREAD_SIZE) {
  local_image[locali+1][localj] = image[k+width];
  }
  else if (locali == 1) {
  local_image[locali-1][localj] = image[k-width];
  }*/

//else if (locali == 1) local_image[locali-1][localj] = image[k-width];
/*if (localj == 16) {//COMPUTE_THREAD_SIZE) {
  local_image[locali][localj+1] = image[k+1];
  }
  else if (localj == 1) {
  local_image[locali][localj-1] = image[k-1];
  }
  else if (locali == COMPUTE_THREAD_SIZE) {
  local_image[locali+1][localj] = image[k+width];
  }
  else if (locali == 1) {
  local_image[locali-1][localj] = image[k-width];
  }*/
__global__ void compute1_shared(double *north_deriv, double *south_deriv, double *west_deriv,
				double *east_deriv, double *diff_coef, unsigned char *image,
				int width, int height, double std_dev)
{
  //extern  __shared__ unsigned char local_image[];
  __shared__ unsigned char local_image[18][18];
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;

  if (j < width && i < height) {
    int locali = threadIdx.y;
    int localj = threadIdx.x;
    long k = i * width + j;
    local_image[locali][localj] = image[k];
    __syncthreads();
    //End of shared data load
    /*int _locali = threadIdx.y + 1;
    int _localj = threadIdx.x + 1;
    int _j = blockDim.x * blockIdx.x + threadIdx.x;
    int _i = blockDim.y * blockIdx.y + threadIdx.y;
    int _k = _i * width + _j;*/
    if(locali > 0 && localj > 0 && j <  blockDim.x-1  && i <  blockDim.y-1){
      unsigned char currentPixel = local_image[locali][localj];
      double ndk, sdk, wdk, edk;
      double coef;
      double laplacian, num, den, std_dev2, gradient_square;
      /*ndk = local_image[(locali-1) * localWidth + localj] - currentPixel;
      sdk = local_image[(locali+1) * localWidth + localj] - currentPixel;
      wdk = local_image[locali * localWidth + localj - 1] - currentPixel;
      edk = local_image[locali * localWidth + localj + 1] - currentPixel;*/
      ndk = local_image[locali-1][localj] - currentPixel;
      sdk = local_image[locali+1][localj] - currentPixel;
      wdk = local_image[locali][localj-1] - currentPixel;
      edk = local_image[locali][localj+1] - currentPixel;
      gradient_square = (ndk * ndk + sdk * sdk + wdk * wdk + edk * edk) / (currentPixel * currentPixel);
      laplacian = (ndk + sdk + wdk + edk) / currentPixel; // 4 doubleing point arithmetic operations
      num = (0.5 * gradient_square) - ((1.0 / 16.0) * (laplacian * laplacian)); // 5 doubleing point arithmetic operations
      den = 1 + (.25 * laplacian); // 2 doubleing point arithmetic operations
      std_dev2 = num / (den * den); // 2 doubleing point arithmetic operations
      den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 doubleing point arithmetic operations
      coef = 1.0 / (1.0 + den); // 2 doubleing point arithmetic operations
      if (coef < 0) {
        diff_coef[k] = 0;
      } else if (coef > 1) {
        diff_coef[k] = 1;
      }
      north_deriv[k] = ndk;
      south_deriv[k] = sdk;
      west_deriv[k] = wdk;
      east_deriv[k] = edk;
    }
  }

}

__global__ void compute2(double *north_deriv, double *south_deriv,
			 double *west_deriv, double *east_deriv, double *diff_coef,
			 unsigned char *image, int width, int height, double lambda)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= 1 && j >= 1 && j < width-1 && i < height-1) {
    int k = i * width + j;
    double diff_coef_north, diff_coef_west, diff_coef_east, diff_coef_south, divergence;
    diff_coef_north = diff_coef[k];    // north diffusion coefficient
    diff_coef_south = diff_coef[k + width];	// south diffusion coefficient
    diff_coef_west = diff_coef_north;	// west diffusion coefficient
    diff_coef_east = diff_coef[k + 1]; // east diffusion coefficient
    divergence = diff_coef_north * north_deriv[k] + diff_coef_south * south_deriv[k] + diff_coef_west * west_deriv[k] + diff_coef_east * east_deriv[k]; // --- 7 doubleing point arithmetic operations
    image[k] = image[k] + 0.25 * lambda * divergence; // --- 3 doubleing point arithmetic operations
  }
}

__global__ void compute2_shared(double *north_deriv, double *south_deriv,
				double *west_deriv, double *east_deriv, double *diff_coef,
				unsigned char *image, int width, int height, double lambda)
{
  //double local_diff_coef[];
  __shared__ double local_diff_coef[18][18];
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if (j < width && i < height) {
    int k = i * width + j;
  /*  int locali = blockIdx.y;
    int localj = blockIdx.x;*/
    int locali = threadIdx.y;
    int localj = threadIdx.x;
    //int localWidth = blockDim.x + 1;
    //int localHeight = blockDim.y + 1;
    //int localk = locali * localWidth + localj;
    local_diff_coef[locali][localj] = diff_coef[k];
    /*if (locali == 16) {
      local_diff_coef[locali+1][localj] = diff_coef[k + width]; //Down tile
      }
      else if (localj == 16) {
      local_diff_coef[locali][localj+1] = diff_coef[k + 1]; //up tile
      }*/
    __syncthreads();
    /*int _j = blockDim.x * blockIdx.x + threadIdx.x;
    int _i = blockDim.y * blockIdx.y + threadIdx.y;
    int _k = _i * width + _j;
    int _locali = blockIdx.y;
    int _localj = blockIdx.x;*/
    if(locali > 0 && localj > 0 && j <  blockDim.x-1  && i <  blockDim.y-1){
      double diff_coef_north, diff_coef_west, diff_coef_east, diff_coef_south, divergence;
      diff_coef_north = local_diff_coef[locali][localj];
      diff_coef_south = local_diff_coef[locali+1][localj];
      diff_coef_west = diff_coef_north;
      diff_coef_east = local_diff_coef[locali][localj+1];
      divergence = diff_coef_north * north_deriv[k] + diff_coef_south * south_deriv[k] + diff_coef_west * west_deriv[k] + diff_coef_east * east_deriv[k]; // --- 7 doubleing point arithmetic operations
      image[k] = image[k] * 0.25 * lambda * divergence;
    }
  }
}

int main(int argc, char *argv[]) {
  // Part I: allocate and initialize variables
  double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;	// time variables
  time_0 = get_time();
  const char *filename = "input.pgm";
  const char *outputname = "output.png";
  int width, height, pixelWidth, n_pixels;
  int n_iter = 50;
  double lambda = 0.5;
  double mean, variance, std_dev;  //local region statistics
  double sum, sum2;	// calculation variables
  time_1 = get_time();

  if(argc<2) {
    printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
    return(-1);
  }
  for(int ac=1;ac<argc;ac++) {
    if(MATCH("-i")) {
      filename = argv[++ac];
    } else if(MATCH("-iter")) {
      n_iter = atoi(argv[++ac]);
    } else if(MATCH("-l")) {
      lambda = atof(argv[++ac]);
    } else if(MATCH("-o")) {
      outputname = argv[++ac];
    } else {
      printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
      return(-1);
    }
  }
  time_2 = get_time();

  // Part III: read image
  printf("Reading image...\n");
  unsigned char *image = stbi_load(filename, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
    return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
  n_pixels = height * width;
  time_3 = get_time();

  // Part IV: allocate variables


  //Part V: compute

  //Data for reduction
  int threads = 1024;
  int blocks = n_pixels / threads;
  double *dimage, *doutput, *d2output;
  double *houtput = (double *)malloc(sizeof(double) * blocks);
  double *h2output = (double *) malloc(sizeof(double) * blocks);
  cudaDeviceReset();
  cudaMalloc(&dimage, sizeof(double) * n_pixels);
  cudaMalloc(&doutput, sizeof(double) * blocks);
  cudaMalloc(&d2output, sizeof(double) * blocks);
  //Data for computes
  double *dnorth_deriv, *dsouth_deriv, *dwest_deriv, *deast_deriv, *ddiff_coefs;
  unsigned char *dimage_uchar;
  cudaMalloc(&dimage_uchar, sizeof(unsigned char) * n_pixels);
  cudaMalloc(&dnorth_deriv, sizeof(double) * n_pixels);
  cudaMalloc(&dsouth_deriv, sizeof(double) * n_pixels);
  cudaMalloc(&dwest_deriv, sizeof(double) * n_pixels);
  cudaMalloc(&deast_deriv, sizeof(double) * n_pixels);
  cudaMalloc(&ddiff_coefs, sizeof(double) * n_pixels);
  cudaMemcpy(dimage_uchar, image, n_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
  dim3 threadsPerBlock(18, 18, 1);
  dim3 blockPerGrid(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
  /*int imgTileSize = 18 * 18;
    int coefTileSize = 17 * 17;*/
  time_4 = get_time();
  for (int iter = 0; iter < n_iter; ++iter) {
    //Reduction
    sumArray<<<blocks, threads, sizeof(double) * threads>>>(doutput, dimage_uchar, n_pixels);
    sum2Array<<<blocks, threads, sizeof(double) * threads>>>(d2output, dimage_uchar, n_pixels);
    cudaDeviceSynchronize();
    cudaMemcpy(houtput, doutput, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(h2output, d2output, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
    sum = sum2 = 0;
    for (int i = 0; i < blocks; i++) {
      sum += houtput[i];
      sum2 += h2output[i];
    }
    mean = sum / n_pixels; // --- 1 doubleing point arithmetic operations
    variance = (sum2 / n_pixels) - mean * mean; // --- 3 doubleing point arithmetic operations
    std_dev = variance / (mean * mean); // --- 2 doubleing point arithmetic operations

    //Compute 1
    /*compute1<<<blockPerGrid, threadsPerBlock>>>(dnorth_deriv, dsouth_deriv, dwest_deriv, deast_deriv,
      ddiff_coefs, dimage_uchar, width, height, std_dev);*/
    compute1_shared<<<blockPerGrid, threadsPerBlock>>>(dnorth_deriv, dsouth_deriv, dwest_deriv, deast_deriv,
						       ddiff_coefs, dimage_uchar, width, height, std_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      {
	fprintf(stderr, "1: Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
      }
    cudaDeviceSynchronize();
    //Compute 2
    /*compute2<<<blockPerGrid, threadsPerBlock>>>(dnorth_deriv, dsouth_deriv, dwest_deriv,
						deast_deriv, ddiff_coefs, dimage_uchar,
						width, height, lambda);*/
    compute2_shared<<<blockPerGrid, threadsPerBlock>>>(dnorth_deriv, dsouth_deriv, dwest_deriv,
      deast_deriv, ddiff_coefs, dimage_uchar,
      width, height, lambda);
    err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess)
      {
	fprintf(stderr, "2: Failed to allocate device vector b (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
      }
  }
  cudaMemcpy(image, dimage_uchar, sizeof(unsigned char) * n_pixels, cudaMemcpyDeviceToHost);
  time_5 = get_time();

  // Part VI: write image to file
  stbi_write_png(outputname, width, height, pixelWidth, image, 0);
  time_6 = get_time();

  // Part VII: get average of sum of pixels for testing and calculate GFLOPS
  // FOR VALIDATION - DO NOT PARALLELIZE
  double test = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      test += image[i * width + j];
    }
  }
  test /= n_pixels;

  double gflops = (double) (n_iter * 1E-9 * (3 * height * width + 42 * (height-1) * (width-1) + 6)) / (time_5 - time_4);
  time_7 = get_time();

  // Part VIII: deallocate variables
  stbi_image_free(image);
  //Clean the GPU memory
  cudaDeviceReset();
  cudaFree(dwest_deriv);
  cudaFree(deast_deriv);
  cudaFree(dsouth_deriv);
  cudaFree(dnorth_deriv);
  cudaFree(ddiff_coefs);
  cudaFree(dimage_uchar);
  cudaFree(doutput);
  cudaFree(d2output);
  //Clean the main memory
  free(houtput);
  free(h2output);

  time_8 = get_time();

  // print
  printf("Time spent in different stages of the application:\n");
  printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
  printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
  printf("%9.6f s => Part III: read image\n", (time_3 - time_2));
  printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
  printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
  printf("%9.6f s => Part VI: write image to file\n", (time_6 - time_5));
  printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
  printf("%9.6f s => Part VIII: deallocate variables\n", (time_7 - time_6));
  printf("Total time: %9.6f s\n", (time_8 - time_0));
  printf("Average of sum of pixels: %9.6f\n", test);
  printf("GFLOPS: %f\n", gflops);
  return 0;
}
