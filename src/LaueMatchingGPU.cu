//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//

#define _XOPEN_SOURCE 500
#include<unistd.h>
#include<cuda.h>
#include "LaueMatchingHeaders.h"

double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;

__global__
void compare(size_t nrPxX, size_t nOr, size_t nrMaxSpots, double minInt, size_t minSps, uint16_t *oA, double *im, double *mA)
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
			thisInt = im[py*nrPxX+px];
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

static inline void usageGPU(){
	puts("LaueMatching on the GPU\n"
	"Contact hsharma@anl.gov\n"
	"Arguments: \n"
	"* 	parameterFile (text)\n"
	"* 	binary files for candidate orientation list [double]\n"
	"* 	text of valid_hkls (preferably sorted on f2), space separated\n"
	"* 	binary file for image [double]\n"
	"* 	number of CPU cores to use: \n"
	"* NOTE: some computers cannot read the full candidate orientation list, \n"
	"* must use multiple cores to distribute in that case\n\n"
	"Parameter file with the following parameters: \n"
	"		* LatticeParameter (in nm and degrees),\n"
	"		* tol_latC (in %, 6 values),\n"
	"		* tol_c_over_a (in %, 1 value), it will only change c, keep a constant,\n"
	"		* SpaceGroup,\n"
	"		* P_Array (3 array describing positioning on the detector),\n"
	"		* R_Array (3 array describing tilts of the detector),\n"
	"		* PxX,\n"
	"		* PxY (PixelSize(x,y)),\n"
	"		* NrPxX,\n"
	"		* NrPxY (NrPixels(x,y)),\n"
	"		* Elo (minimum energy for simulating diffraction spots),\n"
	"		* Ehi (maximum energy for simulating diffraction spots),\n"
	"		* MaxNrLaueSpots(maximum number of spots to simulate),\n"
	"		* ForwardFile (file name to save forward simulation result),\n"
	"		* DoFwd (whether to do forward simulation, ensure ForwardFile exists),\n"
	"		* MinNrSpots (minimum number of spots that qualify a grain, must\n"
	"					  be smaller than MaxNrLaueSpots),\n"
	"		* MinIntensity (minimum total intensity from the MinNrSpots that\n"
	"						will qualify a match, usually 100 counts),\n"
	"		* MaxAngle (maximum angle in degrees that defines a grain,\n" 
	"					if misorientation between two candidates is smaller \n"
	"					than this, the solutions will be merged).\n");
}


int main(int argc, char *argv[])
{
if (argc!=6){
		usageGPU();
		return(0);
	}
	char *paramFN = argv[1];
	FILE *fileParam;
	fileParam = fopen(paramFN,"r");
	char aline[1000], *str, dummy[1000], outfn[1000];
	int LowNr, nrPxX, nrPxY, maxNrSpots=500, minNrSpots=5, doFwd=1;
	sg_num=225;
	double pArr[3], rArr[3], pxX, pxY, Elo = 5, Ehi = 30;
	int iter;
	for (iter=0;iter<6;iter++) tol_LatC[iter] = 0;
	double minIntensity=1000.0,maxAngle=2.0;
	double LatticeParameter[6];
	puts("Reading parameter file");
	fflush(stdout);
	tol_c_over_a = 0;
	while (fgets(aline,1000,fileParam)!=NULL){
		str = "LatticeParameter";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf %lf %lf %lf %lf %lf",dummy,
				&LatticeParameter[0],&LatticeParameter[1],&LatticeParameter[2],
				&LatticeParameter[3],&LatticeParameter[4],&LatticeParameter[5]);
			calcV(LatticeParameter);
			continue;
		}
		str = "P_Array";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf %lf %lf",dummy,&pArr[0],&pArr[1],&pArr[2]);
			continue;
		}
		str = "R_Array";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf %lf %lf",dummy,&rArr[0],&rArr[1],&rArr[2]);
			continue;
		}
		str = "tol_c_over_a";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&tol_c_over_a);
			continue;
		}
		str = "PxX";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&pxX);
			continue;
		}
		str = "PxY";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&pxY);
			continue;
		}
		str = "Elo";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&Elo);
			continue;
		}
		str = "Ehi";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&Ehi);
			continue;
		}
		str = "DoFwd";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&doFwd);
			continue;
		}
		str = "NrPxX";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&nrPxX);
			continue;
		}
		str = "NrPxY";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&nrPxY);
			continue;
		}
		str = "MaxNrLaueSpots";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&maxNrSpots);
			continue;
		}
		str = "MinNrSpots";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&minNrSpots);
			continue;
		}
		str = "SpaceGroup";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %d",dummy,&sg_num);
			continue;
		}
		str = "MinIntensity";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&minIntensity);
			continue;
		}
		str = "MaxAngle";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf",dummy,&maxAngle);
			continue;
		}
		str = "ForwardFile";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %s",dummy,&outfn);
			continue;
		}
		str = "tol_LatC";
		LowNr = strncmp(aline,str,strlen(str));
		if (LowNr==0){
			sscanf(aline,"%s %lf %lf %lf %lf %lf %lf",dummy,&tol_LatC[0],&tol_LatC[1],
								  &tol_LatC[2],&tol_LatC[3],&tol_LatC[4],&tol_LatC[5]);
			continue;
		}
	}
	if (tol_c_over_a!=0){
		tol_LatC[0] = 0;
		tol_LatC[1] = 0;
		tol_LatC[2] = 0;
		tol_LatC[3] = 0;
		tol_LatC[4] = 0;
		tol_LatC[5] = 0;
	}
	c_over_a_orig = LatticeParameter[2]/LatticeParameter[0];
	fclose(fileParam);
	puts("Parameters read");
	
	// Since we always go in the opposite direction (rotate spots to detector)
	// and determinant(rot)=1, we just use the transpose of rot instead of rot_inv.
	double rotang = CalcLength(rArr[0],rArr[1],rArr[2]);
	double rotvect[3] = {rArr[0]/rotang,rArr[1]/rotang,rArr[2]/rotang};
	double rot[3][3] = {{cos(rotang)+(1-cos(rotang))*(rotvect[0]*rotvect[0]), 
					     (1-cos(rotang))*rotvect[0]*rotvect[1]-sin(rotang)*rotvect[2], 
					     (1-cos(rotang))*rotvect[0]*rotvect[2]+sin(rotang)*rotvect[1]},
					    {(1-cos(rotang))*rotvect[1]*rotvect[0]+sin(rotang)*rotvect[2], 
						 cos(rotang)+(1-cos(rotang))*(rotvect[1]*rotvect[1]),  
						 (1-cos(rotang))*rotvect[1]*rotvect[2]-sin(rotang)*rotvect[0]},
						{(1-cos(rotang))*rotvect[2]*rotvect[0]-sin(rotang)*rotvect[1], 
						 (1-cos(rotang))*rotvect[2]*rotvect[1]+sin(rotang)*rotvect[0], 
						 cos(rotang)+(1-cos(rotang))*(rotvect[2]*rotvect[2])}};
	double rotTranspose[3][3] = {{rot[0][0],rot[1][0],rot[2][0]},
								 {rot[0][1],rot[1][1],rot[2][1]},
								 {rot[0][2],rot[1][2],rot[2][2]}};

	// Read orientations from a binary file
	puts("Reading orientations");
    double st_tm = omp_get_wtime();
	fflush(stdout);
	char *orientFN = argv[2];
	FILE *orientF = fopen(orientFN,"rb");
	if (orientF == NULL){
		puts("Could not read orientation file, exiting.");
		return(1);
	}
	fseek(orientF,0L,SEEK_END);
	size_t szFile = ftell(orientF);
	rewind(orientF);
	size_t nrOrients = (size_t)((double)szFile / (double)(9*sizeof(double)));
	double *orients;
	orients = (double *) malloc(szFile);
	fread(orients,1,szFile,orientF);
	fclose(orientF);
	printf("%zu Orientations read, took %lf seconds, now reading hkls\n",nrOrients,omp_get_wtime()-st_tm);
	fflush(stdout);
	
	// Read precomputed hkls from python
	char *hklfn = argv[3];
	FILE *hklf;
	hklf = fopen(hklfn,"r");
	if (hklf == NULL){
		printf("Could not read hkl file %s.\n",hklfn);
		return 1;
	}
	int *hkls;
	hkls = (int *) calloc(MaxNHKLS*3,sizeof(*hkls));
	int nhkls = 0;
	while (fgets(aline,1000,hklf)!=NULL){
		sscanf(aline,"%d %d %d",&hkls[nhkls*3+0],&hkls[nhkls*3+1],&hkls[nhkls*3+2]);
		nhkls++;
	}
	fclose(hklf);
	
	// Read image file
	printf("%'d hkls read. \nNow reading image file %s, number of pixels %d. \nminNrSpots: %d, MinIntensity: %d\n",
			nhkls,argv[4],nrPxX*nrPxY,minNrSpots,(int)minIntensity);
	char *imageFN = argv[4];
	FILE *imageFile;
	imageFile = fopen(imageFN,"rb");
	if (imageFile == NULL){
		printf("Could not read image file %s.\n",imageFN);
		return 1;
	}
	double *image;
	image = (double *) malloc(nrPxX*nrPxY*sizeof(*image));
	fread(image,nrPxX*nrPxY*sizeof(*image),1,imageFile);
	int pxNr,nonZeroPx=0;
	for (pxNr=0;pxNr<nrPxX*nrPxY;pxNr++){
		if (image[pxNr]>0){
			nonZeroPx++;
		}
	}
	fclose(imageFile);
	printf("Pixels with intensity: %d\n",nonZeroPx);

	// Now we have orientations, hkls and all other parameters, let's do forward simulation.
	double *matchedArr;
	matchedArr = (double *) calloc(nrOrients,sizeof(*matchedArr));
	if (matchedArr == NULL){
		printf("Could not allocate matchedArr, requested %zu bytes. Please check.\n",(size_t)(nrOrients));
		return 1;
	}
	int numProcs = atoi(argv[5]);
	printf("Now running using %d threads.\n",numProcs);
	fflush(stdout);
	double ki[3] = {0,0,1.0};
	double start_time = omp_get_wtime();
	int global_iterator,k,l,m;
	size_t nrResults=0;
	double recip[3][3];
	calcRecipArray(LatticeParameter,sg_num,recip);
	if (doFwd==0){
		printf("Trying to see if the forward simulation exists. Looking for %s file.\n",outfn);
		int result = open(outfn, O_RDONLY, S_IRUSR|S_IWUSR);
		if (result<1){
			printf("Could not read the forward simulation file. Running in simulation mode!\n");
			doFwd = 1;
		} else printf("%s file was found. Will not do forward simulation.\n");
		close(result);
	} else printf("Forward simulation was requested, will be saved to %s.\n",outfn);
	
	if (doFwd==0){
		#pragma omp parallel num_threads(numProcs)
		{
			int procNr = omp_get_thread_num();
			int nrOrientsThread = (int)ceil((double)nrOrients/(double)numProcs);
			int startOrientNr = procNr * nrOrientsThread;
			int endOrientNr = startOrientNr + nrOrientsThread;
			if (endOrientNr > nrOrients) endOrientNr = nrOrients;
			nrOrientsThread = endOrientNr - startOrientNr;
			uint16_t *outArrThis;
			size_t szArr = nrOrientsThread*(1+2*maxNrSpots);
			outArrThis = (uint16_t *) calloc(szArr,sizeof(*outArrThis));
			if (outArrThis == NULL){
				printf("Could not allocate outArr per thread, needed %lldMB of RAM. Behavior unexpected now.\n"
					,(long long int)nrOrientsThread*(10+5*maxNrSpots)*sizeof(double)/(1024*1024));
			}
			size_t OffsetHere;
			OffsetHere = procNr;
			OffsetHere *= szArr;
			OffsetHere *= sizeof(*outArrThis);
			if (doFwd == 0){
				int result = open(outfn, O_RDONLY|O_SYNC, S_IRUSR|S_IWUSR);
				ssize_t readBytes = pread(result,outArrThis,szArr*sizeof(*outArrThis),OffsetHere);
				if (readBytes != szArr*sizeof(*outArrThis)){
					// printf("Did not finish reading, going again.\n");
					OffsetHere+=readBytes;
					size_t offset_arr = readBytes / sizeof(*outArrThis);
					size_t bytesRemaining = szArr*sizeof(*outArrThis) - readBytes;
					readBytes = pread(result,outArrThis+offset_arr,bytesRemaining,OffsetHere);
					if (readBytes!=bytesRemaining) printf("Second try didn't work either."
											"Too big array. Update code. Read %zu bytes, but wanted to read %zu bytes.\n",
											readBytes,bytesRemaining);
				}
				close(result);
			}
			int orientNr;
			double *qhatarr;
			qhatarr = (double *) calloc(maxNrSpots*3,sizeof(qhatarr));
			size_t loc;
			uint16_t px,py;
			double thisInt;
			double tO[3][3], thisOrient[3][3];
			int i,j;
			int hklnr, badSpot;
			double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3],xp,yp,sinTheta,E;
			int spotNr, iterNr;
			int nSpots;
			double totInt;
			for (orientNr = startOrientNr; orientNr < endOrientNr; orientNr++){
				nSpots = 0;
				totInt = 0;
				spotNr = 0;
				for (i=0;i<3;i++) for (j=0;j<3;j++) tO[i][j] = orients[orientNr*9+i*3+j];
				MatrixMultF33(tO,recip,thisOrient);
				for (hklnr=0;hklnr<nhkls;hklnr++){
					hkl[0] = hkls[hklnr*3+0];
					hkl[1] = hkls[hklnr*3+1];
					hkl[2] = hkls[hklnr*3+2];
					MatrixMultF(thisOrient,hkl,qvec);
					qlen = CalcLength(qvec[0],qvec[1],qvec[2]);
					if (qlen ==0) continue;
					qhat[0] = qvec[0]/qlen;
					qhat[1] = qvec[1]/qlen;
					qhat[2] = qvec[2]/qlen;
					dot = qhat[2];
					kf[0] = ki[0] - 2*dot*qhat[0];
					kf[1] = ki[1] - 2*dot*qhat[1];
					kf[2] = ki[2] - 2*dot*qhat[2];
					MatrixMultF(rotTranspose,kf,xyz);
					if (xyz[2]<=0) continue;
					xyz[0] = xyz[0]*pArr[2]/xyz[2];
					xyz[1] = xyz[1]*pArr[2]/xyz[2];
					xyz[2] = pArr[2];
					xp = xyz[0]-pArr[0];
					yp = xyz[1]-pArr[1];
					px = (uint16_t)((xp/pxX) + (0.5*(nrPxX-1)));
					if (px <0 || px > (nrPxX-1)) continue;
					py = (uint16_t)((yp/pxY) + (0.5*(nrPxY-1)));
					if (py <0 || py > (nrPxY-1)) continue;
					sinTheta = -qhat[2];
					E = hc_keVnm * qlen / (4*M_PI*sinTheta);
					if (E < Elo || E > Ehi) continue;
					badSpot = 0;
					for (iterNr=0;iterNr<spotNr;iterNr++){
						if ((fabs(qhat[0] - qhatarr[3*iterNr+0])*100000 < 0.1)&&
							(fabs(qhat[1] - qhatarr[3*iterNr+1])*100000 < 0.1)&&
							(fabs(qhat[2] - qhatarr[3*iterNr+2])*100000 < 0.1)) {
								badSpot = 1;
								break;
						}
					}
					if (badSpot == 0){
						qhatarr[3*spotNr+0] = qhat[0];
						qhatarr[3*spotNr+1] = qhat[1];
						qhatarr[3*spotNr+2] = qhat[2];
						outArrThis[(orientNr-startOrientNr)*(1+2*maxNrSpots)+1+2*spotNr+0] = px;
						outArrThis[(orientNr-startOrientNr)*(1+2*maxNrSpots)+1+2*spotNr+1] = py;
						thisInt = image[(int)((int)py*nrPxX+(int)px)];
						if (thisInt >0){
							totInt += thisInt;
							nSpots++;
						}
						spotNr++;
						if (spotNr == maxNrSpots){
							break;
						}
					}
				}
				outArrThis[(orientNr-startOrientNr)*(1+2*maxNrSpots)+0] = (uint16_t)spotNr;
				if (nSpots >= minNrSpots && totInt >= minIntensity){
					#pragma omp critical
					{
						nrResults++;
					}
					matchedArr[orientNr] = totInt * sqrt((double)nSpots);
				}
			}
			size_t OffsetHereOut;
			OffsetHereOut = procNr;
			OffsetHereOut *= szArr;
			OffsetHereOut *= sizeof(*outArrThis);
			int result = open(outfn, O_CREAT|O_WRONLY|O_SYNC, S_IRUSR|S_IWUSR);
			if (result <= 0){
				printf("Could not open output file.\n");
			}
			ssize_t rc = pwrite(result,outArrThis,szArr*sizeof(*outArrThis),OffsetHereOut);
			if (rc < 0) printf("Could not write to output file\n");
			else if (rc != szArr*sizeof(*outArrThis)){
				OffsetHereOut+=rc;
				size_t offset_arr = rc / sizeof(*outArrThis);
				size_t bytesRemaining = szArr*sizeof(*outArrThis) - rc;
				rc = pwrite(result,outArrThis+offset_arr,bytesRemaining,OffsetHereOut);
				if (rc!=bytesRemaining) printf("Second try didn't work either. Too big array. Update code.\n");
			}
			close(result);
			free(outArrThis);
		}
	} else{
		size_t szArr = nrOrients *(1+2*maxNrSpots);
		uint16_t *outArr;
		outArr = (uint16_t *) calloc(szArr,sizeof(uint16_t));
		FILE *fwdFN;
		fwdFN = fopen(outfn,"rb");
		if(outArr==NULL){
			printf("Could not allocate.\n");
			fflush(stdout);
			return 1;
		}
		fseek(fwdFN,0L,SEEK_END);
		size_t readBytes = fread(outArr,szArr*sizeof(uint16_t),1,fwdFN);
		// CUDA BLOCK
		uint16_t *device_outArr;
		cudaMalloc(&device_outArr,szArr*sizeof(uint16_t));
		double *device_image;
		cudaMalloc(&device_image,nrPxX*nrPxY*sizeof(double));
		double *device_matchedArr, *mArr;
		cudaMalloc(&device_matchedArr,nrOrients*sizeof(double));
		mArr = (double *) malloc(nrOrients*sizeof(double));
		cudaMemcpy(device_outArr,outArr,szArr*sizeof(uint16_t),cudaMemcpyHostToDevice);
		clock_t start = clock();
		cudaMemcpy(device_image,image,nrPxX*nrPxY*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemset(device_matchedArr,0,nrOrients*sizeof(double));
		compare<<<(nrOrients+1023)/1024, 1024>>>(nrPxX,nrOrients,maxNrSpots,minIntensity,minNrSpots,device_outArr,device_image,device_matchedArr);
		cudaDeviceSynchronize();
		cudaMemcpy(mArr,device_matchedArr,nrOrients*sizeof(double),cudaMemcpyDeviceToHost);
		clock_t end = clock();
		printf("%lf\n",(double)(end-start)/CLOCKS_PER_SEC);
		size_t nrMatches=0;
		for (int i=0;i<nrOrients;i++){
			if (mArr[i]>0){
				matchedArr[nrMatches] = mArr[i];
				nrMatches++;
			}
		}
		printf("NrMatches: %zu\n",nrMatches);
		nrResults = nrMatches;

	}
	double time2 = omp_get_wtime() - start_time;
	printf("Finished comparing, time elapsed after comparing with forward simulation: %lf seconds.\n"
		"Searching for unique solutions.\n",time2);
	fflush(stdout);
	
	// Figure out the unique orientations (within maxAngle) and do optimization for those.
	// This is not parallelized because we normally don't have many of these.
	// Look into making this parallel.
	double orient1[9], orient2[9], quat1[4], quat2[4], misoAngle, bestIntensity;
	double orientBest[3][3], eulerBest[3],eulerFit[3], orientFit[3][3], bestOverlap;
	double tol= 3*deg2rad;
	// Check if we can increase maxNrSpots and see if we find more spots
	maxNrSpots *= 3;
	int *doneArr;
	doneArr = (int *) calloc(nrOrients,sizeof(*doneArr));
	int unique = 0, bestSol;
	char outFN[1000];
	sprintf(outFN,"%s.solutions.txt",imageFN);
	FILE *outF = fopen(outFN,"w");
	if (outF==NULL){
		printf("Could not open file from writing solutions Exiting\n");
		return(1);
	}
	sprintf(outFN,"%s.spots.txt",imageFN);
	FILE *ExtraInfo = fopen(outFN,"w");
	fprintf(ExtraInfo,"%%GrainNr\tSpotNr\th\tk\tl\tX\tY\tQhat[0]\tQhat[1]\tQhat[2]\tIntensity\n");
	fprintf(outF,"%%GrainNr\tNumberOfSolutions\tIntensity\tNMatches*Intensity\tNMatches*sqrt(Intensity)\t"
		"NMatches\tNSpotsCalc\t""Recip1\tRecip2\tRecip3\tRecip4\tRecip5\tRecip6\tRecip7\tRecip8\tRecip9\t"
		"LatticeParameterFit[a]\tLatticeParameterFit[b]\tLatticeParameterFit[c]\t"
		"LatticeParameterFit[alpha]\tLatticeParameterFit[beta]\tLatticeParameterFit[gamma]\t"
		"OrientMatrix0\tOrientMatrix1\tOrientMatrix2\tOrientMatrix3\tOrientMatrix4\tOrientMatrix5\t"
		"OrientMatrix6\tOrientMatrix7\tOrientMatrix8\t"
		"CoarseNMatches*sqrt(Intensity)\t""misOrientationPostRefinement[degrees]\torientationRowNr\n");
	// Make an array with the orientations to process
	double *FinOrientArr;
	FinOrientArr = (double *) calloc(nrResults*9,sizeof(*FinOrientArr));
	int iterNr = 0;
	int *dArr, *bsArr;
	dArr = (int *) calloc(nrResults,sizeof(*dArr));
	bsArr = (int *) calloc(nrResults,sizeof(*bsArr));
	for (global_iterator=0;global_iterator<nrOrients;global_iterator++){
		if (matchedArr[global_iterator]==0) continue;
		if (doneArr[global_iterator] != 0) continue;
		for (k=0;k<9;k++){
			orient1[k] = orients[global_iterator*9+k];
		}
		OrientMat2Quat(orient1,quat1);
		doneArr[global_iterator] ++;
		bestSol = global_iterator;
		bestIntensity = matchedArr[global_iterator];
		for (l=global_iterator+1;l<nrOrients;l++){
			if (matchedArr[l]==0) continue;
			if (doneArr[l] > 0) continue;
			for (m=0;m<9;m++){
				orient2[m] = orients[l*9+m];
			}
			OrientMat2Quat(orient2,quat2);
			misoAngle = GetMisOrientation(quat1,quat2,sg_num);
			if (misoAngle <= maxAngle) {
				doneArr[l] = 1;
				doneArr[global_iterator] ++;
				if (matchedArr[l] > bestIntensity){
					bestIntensity = matchedArr[l];
					bestSol = l;
				}
			}
		}
		for (k=0;k<9;k++){
			FinOrientArr[iterNr*9+k] = orients[bestSol*9+k];
		}
		dArr[iterNr] = doneArr[global_iterator];
		bsArr[iterNr] = bestSol;
		iterNr ++;
	}
	int totalSols = iterNr;
	# pragma omp parallel for num_threads(numProcs)
	for (iterNr=0;iterNr<totalSols;iterNr++){
		double orientBest[3][3], eulerBest[3],eulerFit[3], orientFit[3][3], bestOverlap, q1[4],q2[4];
		int iJ, iK, iL;
		for (iJ=0;iJ<3;iJ++){
			for (iK=0;iK<3;iK++){
				orientBest[iJ][iK] = FinOrientArr[iterNr*9 + 3*iJ + iK];
			}
		}
		OrientMat2Euler(orientBest,eulerBest);
		for (iJ=0;iJ<3;iJ++){
			eulerFit[iJ] = eulerBest[iJ];
		}
		int saveExtraInfo = 0;
		int doCrystalFit = 0;
		double *outArrThisFit;
		outArrThisFit = (double *) calloc(3*maxNrSpots,sizeof(*outArrThisFit));
		double latCFit[6],recipFit[3][3],mv=0;
		FitOrientation(image,eulerBest,hkls,nhkls,nrPxX,nrPxY,recip,outArrThisFit,maxNrSpots,
			rotTranspose,pArr,pxX,pxY,Elo,Ehi,tol,LatticeParameter,eulerFit,latCFit,&mv, doCrystalFit);
		doCrystalFit = 1;
		for (iK=0;iK<3;iK++) eulerBest[iK] = eulerFit[iK];
		FitOrientation(image,eulerBest,hkls,nhkls,nrPxX,nrPxY,recip,outArrThisFit,maxNrSpots,
			rotTranspose,pArr,pxX,pxY,Elo,Ehi,tol,LatticeParameter,eulerFit,latCFit,&mv,doCrystalFit);
		Euler2OrientMat(eulerFit,orientFit);
		OrientMat2Quat33(orientBest,q1);
		OrientMat2Quat33(orientFit,q2);
		int simulNrSps=0;
		calcRecipArray(latCFit,sg_num,recipFit);
		int nrSps = writeCalcOverlap(image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit, outArrThisFit, maxNrSpots, 
			rotTranspose, pArr, pxX, pxY, Elo, Ehi, ExtraInfo, saveExtraInfo, &simulNrSps);
		if (nrSps>=minNrSpots){
			int bs = bsArr[iterNr];
			double miso = GetMisOrientation(q1,q2,sg_num);
			saveExtraInfo = iterNr+1;
			calcRecipArray(latCFit,sg_num,recipFit);
			writeCalcOverlap(image,eulerFit,hkls,nhkls,nrPxX,nrPxY,recipFit,outArrThisFit,
				maxNrSpots,rotTranspose,pArr,pxX,pxY,Elo,Ehi,ExtraInfo,saveExtraInfo,&simulNrSps);
			double OF[3][3];
			MatrixMultF33(orientFit,recipFit,OF);
			# pragma omp critical
			{
				fprintf(outF,"%d\t%d\t",iterNr+1,dArr[iterNr]);
				fprintf(outF,"%-13.4lf\t",(mv/nrSps)*(mv/nrSps));
				fprintf(outF,"%-13.4lf\t",nrSps*(mv/nrSps)*(mv/nrSps));
				fprintf(outF,"%-13.4lf\t",mv);
				fprintf(outF,"%d\t",nrSps);
				fprintf(outF,"%d\t",simulNrSps);
				for (k=0;k<3;k++){
					for (l=0;l<3;l++){
						fprintf(outF,"%-13.7lf\t\t",OF[k][l]);
					}
				}
				for (k=0;k<6;k++) fprintf(outF,"%-13.7lf\t\t",latCFit[k]);
				for (k=0;k<3;k++){
					for (l=0;l<3;l++){
						fprintf(outF,"%-13.7lf\t\t",orientFit[k][l]);
					}
				}
				fprintf(outF,"%-13.4lf\t%-13.7lf\t%d\n",matchedArr[bs],miso,bs);
			}
		}
	}
	fclose(ExtraInfo);
	fclose(outF);
	double time = omp_get_wtime() - start_time - time2;
	printf("Finished, time elapsed in fitting: %lf seconds.\n"
		"Initial solutions: %d Unique Orientations: %d\n",time,nrResults,totalSols);

}