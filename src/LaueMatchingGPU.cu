#include<stdio.h>
#include<time.h>
#include<malloc.h>
#include<stdint.h>
#include<math.h>
#include<omp.h>
#include<cuda.h>
#include<fcntl.h>

size_t MaxNrSpots;
size_t nrPixels;

__global__
void compare(size_t nrPx, size_t nOr, size_t nrMaxSpots, double minInt, size_t minSps, uint16_t *oA, double *im, double *mA)
{
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nOr){
		size_t loc = i*(1+2*nrMaxSpots);
		size_t nrSpots = (size_t) oA[loc];
		size_t hklnr;
		size_t px,py;
		double thisInt, totInt=0;
		size_t nSps = 0;
		for (hklnr=0;hklnr<nrSpots;hklnr++){
			loc++;
			px = (size_t) oA[loc];
			loc++;
			py = (size_t) oA[loc];
			thisInt = im[py*nrPx+px];
			if (thisInt>0){
				totInt += thisInt;
				nSps++;
			}
		}
		if (nSps>=minSps && totInt>=minInt){
			mA[i] = totInt*sqrt((double)nSps);
		}
	}
}

int main(int argc, char *argv[])
{
	nrPixels = 2048;
	// MaxNrSpots = 30;
	// size_t minSps = 7;
	// double minInt = 100;
	// FILE *fwdFN = fopen("orientation_files/225_Ni/compact_Ni_FwdSim.bin","rb");
	// FILE *imFN = fopen("results_Ni_cleaned/test_4.h5.bin","rb");
	MaxNrSpots = 180;
	size_t minSps = 35;
	double minInt = 300;
	FILE *fwdFN = fopen("orientation_files/4_Eu2AlO4/Eu2AlO4_FwdSim.bin","rb");
	FILE *imFN = fopen("results_EuAl2O4_Modified_AllowedToMove/test_6_cleaned.h5.bin","rb");
	FILE *orientF = fopen("orientation_files/100MilOrients.bin","rb");
	fseek(orientF,0L,SEEK_END);
	size_t szFile = ftell(orientF);
	rewind(orientF);
	size_t nrOrients = (size_t)((double)szFile / (double)(9*sizeof(double)));
	double *orients;
	orients = (double *) malloc(szFile);
	fread(orients,1,szFile,orientF);
	fclose(orientF);
	// All we need are fwdsim.bin, image.bin and we create an array with results.
	size_t szArr = nrOrients *(1+2*MaxNrSpots);
	uint16_t *outArr;
	outArr = (uint16_t *) calloc(szArr,sizeof(uint16_t));
	if(outArr==NULL){
		printf("Could not allocate.\n");
		fflush(stdout);
		return 1;
	}
	fseek(fwdFN,0L,SEEK_END);
	size_t sz = ftell(fwdFN);
	rewind(fwdFN);
	size_t readBytes = fread(outArr,szArr*sizeof(uint16_t),1,fwdFN);
	fclose(fwdFN);
	if (imFN == NULL) return 1;
	double *image;
	image = (double *) malloc(nrPixels*nrPixels*sizeof(*image));
	readBytes = fread(image,nrPixels*nrPixels*sizeof(*image),1,imFN);
	fclose(imFN);
	uint16_t *device_outArr;
	cudaMalloc(&device_outArr,szArr*sizeof(uint16_t));
	double *device_image;
	cudaMalloc(&device_image,nrPixels*nrPixels*sizeof(double));
	double *device_matchedArr, *matchedArr;
	cudaMalloc(&device_matchedArr,nrOrients*sizeof(double));
	matchedArr = (double *) malloc(nrOrients*sizeof(double));
	cudaMemcpy(device_outArr,outArr,szArr*sizeof(uint16_t),cudaMemcpyHostToDevice);
	clock_t start = clock();
	cudaMemcpy(device_image,image,nrPixels*nrPixels*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemset(device_matchedArr,0,nrOrients*sizeof(double));
	compare<<<(nrOrients+1023)/1024, 1024>>>(nrPixels,nrOrients,MaxNrSpots,minInt,minSps,device_outArr,device_image,device_matchedArr);
	cudaDeviceSynchronize();
	cudaMemcpy(matchedArr,device_matchedArr,nrOrients*sizeof(double),cudaMemcpyDeviceToHost);
	clock_t end = clock();
	printf("%lf\n",(double)(end-start)/CLOCKS_PER_SEC);
	size_t nrMatches=0;
	for (int i=0;i<nrOrients;i++){
		if (matchedArr[i]>0){
			nrMatches++;
		}
	}
	printf("NrMatches: %zu\n",nrMatches);
}