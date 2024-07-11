//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//

#define _XOPEN_SOURCE 500
#include<unistd.h>
#include<cuda.h>
extern "C" {
#include "LaueMatchingHeaders.h"
}

double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;

inline double sin_cos_to_angle (double s, double c){return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);}

inline void normalizeQuat(double quat[4]){
	double norm = sqrt(quat[0]*quat[0]+quat[1]*quat[1]+quat[2]*quat[2]+quat[3]*quat[3]);
	quat[0] /= norm;
	quat[1] /= norm;
	quat[2] /= norm;
	quat[3] /= norm;
}

double TricSym[2][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {1.00000,   0.00000,   0.00000,   0.00000}};

double MonoSym[2][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.00000,   1.00000,   0.00000,   0.00000}};

double OrtSym[4][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {1.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.00000,   0.00000,   0.00000,   1.00000}};

double TetSym[8][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.70711,   0.00000,   0.00000,   0.70711},
   {0.00000,   0.00000,   0.00000,   1.00000},
   {0.70711,  -0.00000,  -0.00000,  -0.70711},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.00000,   0.70711,   0.70711,   0.00000},
   {0.00000,  -0.70711,   0.70711,   0.00000}};

double TrigSym[6][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.50000,   0.00000,   0.00000,   0.86603},
   {0.50000,  -0.00000,  -0.00000,  -0.86603},
   {0.00000,   0.50000,  -0.86603,   0.00000},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.50000,   0.86603,   0.00000}};

double HexSym[12][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.86603,   0.00000,   0.00000,   0.50000},
   {0.50000,   0.00000,   0.00000,   0.86603},
   {0.00000,   0.00000,   0.00000,   1.00000},
   {0.50000,  -0.00000,  -0.00000,  -0.86603},
   {0.86603,  -0.00000,  -0.00000,  -0.50000},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.00000,   0.86603,   0.50000,   0.00000},
   {0.00000,   0.50000,   0.86603,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.00000,  -0.50000,   0.86603,   0.00000},
   {0.00000,  -0.86603,   0.50000,   0.00000}};

double CubSym[24][4] = {
   {1.00000,   0.00000,   0.00000,   0.00000},
   {0.70711,   0.70711,   0.00000,   0.00000},
   {0.00000,   1.00000,   0.00000,   0.00000},
   {0.70711,  -0.70711,   0.00000,   0.00000},
   {0.70711,   0.00000,   0.70711,   0.00000},
   {0.00000,   0.00000,   1.00000,   0.00000},
   {0.70711,   0.00000,  -0.70711,   0.00000},
   {0.70711,   0.00000,   0.00000,   0.70711},
   {0.00000,   0.00000,   0.00000,   1.00000},
   {0.70711,   0.00000,   0.00000,  -0.70711},
   {0.50000,   0.50000,   0.50000,   0.50000},
   {0.50000,  -0.50000,  -0.50000,  -0.50000},
   {0.50000,  -0.50000,   0.50000,   0.50000},
   {0.50000,   0.50000,  -0.50000,  -0.50000},
   {0.50000,   0.50000,  -0.50000,   0.50000},
   {0.50000,  -0.50000,   0.50000,  -0.50000},
   {0.50000,  -0.50000,  -0.50000,   0.50000},
   {0.50000,   0.50000,   0.50000,  -0.50000},
   {0.00000,   0.70711,   0.70711,   0.00000},
   {0.00000,  -0.70711,   0.70711,   0.00000},
   {0.00000,   0.70711,   0.00000,   0.70711},
   {0.00000,   0.70711,   0.00000,  -0.70711},
   {0.00000,   0.00000,   0.70711,   0.70711},
   {0.00000,   0.00000,   0.70711,  -0.70711}};

inline
void QuaternionProduct(double q[4], double r[4], double Q[4])
{
	Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3];
	Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3];
	Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1];
	Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2];
	if (Q[0] < 0) {
		Q[0] = -Q[0];
		Q[1] = -Q[1];
		Q[2] = -Q[2];
		Q[3] = -Q[3];
	}
	normalizeQuat(Q);
}

inline
int MakeSymmetries(int SGNr, double Sym[24][4])
{
	int i, j, NrSymmetries;;
	if (SGNr <= 2){ // Triclinic
		NrSymmetries = 1;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TricSym[i][j];
			}
		}
	}else if (SGNr > 2 && SGNr <= 15){  // Monoclinic
		NrSymmetries = 2;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = MonoSym[i][j];
			}
		}
	}else if (SGNr >= 16 && SGNr <= 74){ // Orthorhombic
		NrSymmetries = 4;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = OrtSym[i][j];
			}
		}
	}else if (SGNr >= 75 && SGNr <= 142){  // Tetragonal
		NrSymmetries = 8;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TetSym[i][j];
			}
		}
	}else if (SGNr >= 143 && SGNr <= 167){ // Trigonal
		NrSymmetries = 6;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TrigSym[i][j];
			}
		}
	}else if (SGNr >= 168 && SGNr <= 194){ // Hexagonal
		NrSymmetries = 12;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = HexSym[i][j];
			}
		}
	}else if (SGNr >= 195 && SGNr <= 230){ // Cubic
		NrSymmetries = 24;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = CubSym[i][j];
			}
		}
	}
	return NrSymmetries;
}

inline
void BringDownToFundamentalRegion(double QuatIn[4], double QuatOut[4],int SGNr)
{
	int i, j, maxCosRowNr=0, NrSymmetries;
	double Sym[24][4];
	if (SGNr <= 2){ // Triclinic
		NrSymmetries = 1;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TricSym[i][j];
			}
		}
	}else if (SGNr > 2 && SGNr <= 15){  // Monoclinic
		NrSymmetries = 2;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = MonoSym[i][j];
			}
		}
	}else if (SGNr >= 16 && SGNr <= 74){ // Orthorhombic
		NrSymmetries = 4;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = OrtSym[i][j];
			}
		}
	}else if (SGNr >= 75 && SGNr <= 142){  // Tetragonal
		NrSymmetries = 8;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TetSym[i][j];
			}
		}
	}else if (SGNr >= 143 && SGNr <= 167){ // Trigonal
		NrSymmetries = 6;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = TrigSym[i][j];
			}
		}
	}else if (SGNr >= 168 && SGNr <= 194){ // Hexagonal
		NrSymmetries = 12;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = HexSym[i][j];
			}
		}
	}else if (SGNr >= 195 && SGNr <= 230){ // Cubic
		NrSymmetries = 24;
		for (i=0;i<NrSymmetries;i++){
			for (j=0;j<4;j++){
				Sym[i][j] = CubSym[i][j];
			}
		}
	}
	double qps[NrSymmetries][4], q2[4], qt[4], maxCos=-10000;
	for (i=0;i<NrSymmetries;i++){
		q2[0] = Sym[i][0];
		q2[1] = Sym[i][1];
		q2[2] = Sym[i][2];
		q2[3] = Sym[i][3];
		QuaternionProduct(QuatIn,q2,qt);
		qps[i][0] = qt[0];
		qps[i][1] = qt[1];
		qps[i][2] = qt[2];
		qps[i][3] = qt[3];
		if (maxCos < qt[0]){
			maxCos = qt[0];
			maxCosRowNr = i;
		}
	}
	QuatOut[0] = qps[maxCosRowNr][0];
	QuatOut[1] = qps[maxCosRowNr][1];
	QuatOut[2] = qps[maxCosRowNr][2];
	QuatOut[3] = qps[maxCosRowNr][3];
	normalizeQuat(QuatOut);
}

inline
double GetMisOrientation(double quat1[4], double quat2[4], int SGNr)
{
	double q1FR[4], q2FR[4], q1Inv[4], QP[4], MisV[4];
	BringDownToFundamentalRegion(quat1,q1FR,SGNr);
	BringDownToFundamentalRegion(quat2,q2FR,SGNr);
	q1Inv[0] = -q1FR[0];
	q1Inv[1] =  q1FR[1];
	q1Inv[2] =  q1FR[2];
	q1Inv[3] =  q1FR[3];
	QuaternionProduct(q1Inv,q2FR,QP);
	BringDownToFundamentalRegion(QP,MisV,SGNr);
	if (MisV[0] > 1) MisV[0] = 1;
	double angle = 2*(acos(MisV[0]))*rad2deg;
	return angle;
}

inline 
void OrientMat2Quat(double OrientMat[9], double Quat[4]){
	double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
	if(trace > 0){
		double s = 0.5/sqrt(trace+1.0);
		Quat[0] = 0.25/s;
		Quat[1] = (OrientMat[7]-OrientMat[5])*s;
		Quat[2] = (OrientMat[2]-OrientMat[6])*s;
		Quat[3] = (OrientMat[3]-OrientMat[1])*s;
	}else{
		if (OrientMat[0]>OrientMat[4] && OrientMat[0]>OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8]);
			Quat[0] = (OrientMat[7]-OrientMat[5])/s;
			Quat[1] = 0.25*s;
			Quat[2] = (OrientMat[1]+OrientMat[3])/s;
			Quat[3] = (OrientMat[2]+OrientMat[6])/s;
		} else if (OrientMat[4] > OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8]);
			Quat[0] = (OrientMat[2]-OrientMat[6])/s;
			Quat[1] = (OrientMat[1]+OrientMat[3])/s;
			Quat[2] = 0.25*s;
			Quat[3] = (OrientMat[5]+OrientMat[7])/s;
		} else {
			double s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4]);
			Quat[0] = (OrientMat[3]-OrientMat[1])/s;
			Quat[1] = (OrientMat[2]+OrientMat[6])/s;
			Quat[2] = (OrientMat[5]+OrientMat[7])/s;
			Quat[3] = 0.25*s;
		}
	}
	if (Quat[0] < 0){
		Quat[0] = -Quat[0];
		Quat[1] = -Quat[1];
		Quat[2] = -Quat[2];
		Quat[3] = -Quat[3];
	}
	normalizeQuat(Quat);
}

inline 
void OrientMat2Quat33(double OM[3][3], double Quat[4]){
	double OrientMat[9];
	int i,j;
	for (i=0;i<3;i++) for (j=0;j<3;j++) OrientMat[i*3+j] = OM[i][j];
	double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
	if(trace > 0){
		double s = 0.5/sqrt(trace+1.0);
		Quat[0] = 0.25/s;
		Quat[1] = (OrientMat[7]-OrientMat[5])*s;
		Quat[2] = (OrientMat[2]-OrientMat[6])*s;
		Quat[3] = (OrientMat[3]-OrientMat[1])*s;
	}else{
		if (OrientMat[0]>OrientMat[4] && OrientMat[0]>OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8]);
			Quat[0] = (OrientMat[7]-OrientMat[5])/s;
			Quat[1] = 0.25*s;
			Quat[2] = (OrientMat[1]+OrientMat[3])/s;
			Quat[3] = (OrientMat[2]+OrientMat[6])/s;
		} else if (OrientMat[4] > OrientMat[8]){
			double s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8]);
			Quat[0] = (OrientMat[2]-OrientMat[6])/s;
			Quat[1] = (OrientMat[1]+OrientMat[3])/s;
			Quat[2] = 0.25*s;
			Quat[3] = (OrientMat[5]+OrientMat[7])/s;
		} else {
			double s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4]);
			Quat[0] = (OrientMat[3]-OrientMat[1])/s;
			Quat[1] = (OrientMat[2]+OrientMat[6])/s;
			Quat[2] = (OrientMat[5]+OrientMat[7])/s;
			Quat[3] = 0.25*s;
		}
	}
	if (Quat[0] < 0){
		Quat[0] = -Quat[0];
		Quat[1] = -Quat[1];
		Quat[2] = -Quat[2];
		Quat[3] = -Quat[3];
	}
	normalizeQuat(Quat);
}


inline
void OrientMat2Euler(double m[3][3],double Euler[3])
{
    double psi, phi, theta, sph;

	if (fabs(m[2][2] - 1.0) < EPS){
		phi = 0;
	}else{
	    phi = acos(m[2][2]);
	}
	sph = sin(phi);
    if (fabs(sph) < EPS)
    {
        psi = 0.0;
        theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0]) : sin_cos_to_angle(-m[1][0], m[0][0]);
    } else{
        psi = (fabs(-m[1][2] / sph) <= 1.0) ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph) : sin_cos_to_angle(m[0][2] / sph,1);
        theta = (fabs(m[2][1] / sph) <= 1.0) ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph) : sin_cos_to_angle(m[2][0] / sph,1);
    }
    Euler[0] = psi;
    Euler[1] = phi;
    Euler[2] = theta;
}

inline void Euler2OrientMat(double Euler[3], double m_out[3][3]){
	double psi = Euler[0];
	double phi = Euler[1];
	double theta = Euler[2];
	double cps = cos(psi);
	double cph = cos(phi);
	double cth = cos(theta);
	double sps = sin(psi);
	double sph = sin(phi);
	double sth = sin(theta);
	m_out[0][0] = cth * cps - sth * cph * sps;
	m_out[0][1] = -cth * cph * sps - sth * cps;
	m_out[0][2] = sph * sps;
	m_out[1][0] = cth * sps + sth * cph * cps;
	m_out[1][1] = cth * cph * cps - sth * sps;
	m_out[1][2] = -sph * cps;
	m_out[2][0] = sth * sph;
	m_out[2][1] = cth * sph;
	m_out[2][2] = cph;
}


inline void MatrixMultF33(double m[3][3],double n[3][3],double res[3][3])
{
	int r;
	for (r=0; r<3; r++) {
		res[r][0] = m[r][0]*n[0][0] +
					m[r][1]*n[1][0] +
					m[r][2]*n[2][0];
		res[r][1] = m[r][0]*n[0][1] +
					m[r][1]*n[1][1] +
					m[r][2]*n[2][1];
		res[r][2] = m[r][0]*n[0][2] +
					m[r][1]*n[1][2] +
					m[r][2]*n[2][2];
	}
}

inline double zeroOut(double val){
	return (fabs(val) < EPS) ? 0 : val;
}


inline void MatrixMultF(double m[3][3],double v[3],double r[3])
{
	int i,j;
	r[0] = 0;
	r[1] = 0;
	r[2] = 0;
	for (i=0; i<3; i++)
		for (j=0;j<3;j++)
			r[i] += m[i][j]*v[j];
}


inline void calcV(double LatC[6]){
	double ca = cos(LatC[3]*deg2rad);
	double cb = cos(LatC[4]*deg2rad);
	double cg = cos(LatC[5]*deg2rad);
	phiVol = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
	cellVol = LatC[0]*LatC[1]*LatC[2]*phiVol;
}

inline void calcRecipArray(double Lat[6], int SpaceGroup, double recip[3][3]){
	double a,b,c,alpha,beta,gamma;
	a = Lat[0]; b = Lat[1]; c= Lat[2]; alpha = Lat[3]; beta = Lat[4], gamma = Lat[5];
	int rhomb = 0;
	if (SpaceGroup == 146 || SpaceGroup == 148 || SpaceGroup == 155 || SpaceGroup == 160 ||
		SpaceGroup == 161 || SpaceGroup == 166 || SpaceGroup == 167) rhomb = 1;
	double ca = cos((alpha)* deg2rad);
	double cb = cos((beta)* deg2rad);
	double cg = cos((gamma)* deg2rad);
	double sg = sin((gamma)* deg2rad);
	double phi = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
	double Vc = a*b*c * phi;
	double pv = (2*M_PI) / (Vc);
	double a0,a1,a2,b0,b1,b2,c0,c1,c2;
	if (rhomb == 0){
		a0 = a			; a1 = 0.0				; a2 = 0.0;
		b0 = b*cg		; b1 = b*sg				; b2=0;
		c0 = c*cb		; c1 = c*(ca-cb*cg)/sg	; c2=c*phi/sg;
		a0 = zeroOut(a0); a1 = zeroOut(a1); a2 = zeroOut(a2);
		b1 = zeroOut(b1); b2 = zeroOut(b2);
		c2 = zeroOut(c2);
	}else{
		double p = sqrt(1.0 + 2*ca);
		double q = sqrt(1.0 - ca);
		double pmq = (a/3.0)*(p-q);
		double p2q = (a/3.0)*(p+2*q);
		a0 = p2q; a1 = pmq; a2 = pmq;
		b0 = pmq; b1 = p2q; b2 = pmq;
		c0 = pmq; c1 = pmq; c2 = p2q;
	}
	recip[0][0] = zeroOut((b1*c2-b2*c1)*pv);
	recip[1][0] = zeroOut((b2*c0-b0*c2)*pv);
	recip[2][0] = zeroOut((b0*c1-b1*c0)*pv);
	recip[0][1] = zeroOut((c1*a2-c2*a1)*pv);
	recip[1][1] = zeroOut((c2*a0-c0*a2)*pv);
	recip[2][1] = zeroOut((c0*a1-c1*a0)*pv);
	recip[0][2] = zeroOut((a1*b2-a2*b1)*pv);
	recip[1][2] = zeroOut((a2*b0-a0*b2)*pv);
	recip[2][2] = zeroOut((a0*b1-a1*b0)*pv);
}

inline double calcOverlap(double *image, double euler[3], int *hkls, int nhkls, int nrPxX, int nrPxY,
	double recip[3][3], double *outArrThis, int maxNrSpots, double rotTranspose[3][3], double pArr[3], double pxX,
	double pxY, double Elo, double Ehi){
	double OM[3][3], OMt[3][3];
	Euler2OrientMat(euler,OMt);
	int i,j;
	double ki[3] = {0,0,1.0};
	MatrixMultF33(OMt,recip,OM);
	int hklnr, badSpot;
	double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3],xp,yp,px,py,sinTheta,E, result=0;
	int spotNr = 0, iterNr, nrPos=0;
	// Save hkl[3], qhat[3], px, py, intensity, spotNr to an array.
	for (hklnr=0;hklnr<nhkls;hklnr++){
		hkl[0] = hkls[hklnr*3+0];
		hkl[1] = hkls[hklnr*3+1];
		hkl[2] = hkls[hklnr*3+2];
		MatrixMultF(OM,hkl,qvec);
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
		px = (xp/pxX) + (0.5*(nrPxX-1));
		if (px <0 || px > (nrPxX-1)) continue;
		py = (yp/pxY) + (0.5*(nrPxY-1));
		if (py <0 || py > (nrPxY-1)) continue;
		sinTheta = -qhat[2];
		E = hc_keVnm * qlen / (4*M_PI*sinTheta);
		if (E < Elo || E > Ehi) continue;
		badSpot = 0;
		for (iterNr=0;iterNr<spotNr;iterNr++){
			if ((fabs(qhat[0] - outArrThis[3*iterNr+0])*100000 < 0.1)&&
			   (fabs(qhat[1] - outArrThis[3*iterNr+1])*100000 < 0.1)&&
			   (fabs(qhat[2] - outArrThis[3*iterNr+2])*100000 < 0.1)) {
					badSpot = 1;
					break;
			   }
		}
		if (badSpot == 0){
			outArrThis[3*spotNr+0] = qhat[0];
			outArrThis[3*spotNr+1] = qhat[1];
			outArrThis[3*spotNr+2] = qhat[2];
			if (image[(int)((int)py*nrPxX+(int)px)] >0){
				result += image[(int)((int)py*nrPxX+(int)px)];
				nrPos++;
			}
			spotNr++;
			if (spotNr == maxNrSpots){
				break;
			}
		}
	}
	result = nrPos*sqrt(result);
	return result;
}

inline double problem_function(unsigned n, const double *x, double *grad, void* f_data_supplied)
{
	int i, j;
	struct dataFit *f_data = (struct dataFit *) f_data_supplied;
	double rotTranspose[3][3], pArr[3], recip[3][3];
	for (i=0;i<3;i++){
		pArr[i] = f_data->pArr[i];
		for (j=0;j<3;j++){
			rotTranspose[i][j] = f_data->rotTranspose[i][j];
			if (n==3) recip[i][j] = f_data->recip[i][j];
		}
	}
	if (n>3){
		// Read the lattice parameter from x and then remake recip
		// We use tol_LatC to know which parts of lattice parameter to fit.
		double latCNew[6];
		int cntr = 0;
		for (i=0;i<6;i++){
			if (tol_LatC[i]!=0){
				latCNew[i] = x[3+cntr];
				cntr++;
			} else{
				latCNew[i] = f_data->LatCOrig[i];
			}
		}
		// Change latCNew so that V is conserved for changing c/a Hexagonal.
		if (tol_c_over_a!=0){
			double aNew = pow((cellVol/(x[3]*phiVol)),1/3);
			latCNew[0] = aNew;
			latCNew[1] = aNew;
			latCNew[2] = x[3]*aNew;
		}
		calcRecipArray(latCNew,sg_num,recip);
	}
	double *image, *outArrThis;
	int *hkls;
	image = &(f_data->image[0]);
	hkls = &(f_data->hkls[0]);
	outArrThis = &(f_data->outArrThis[0]);
	double overlap = 0;
    double Euler[3];
    for (i=0;i<3;i++) Euler[i] = x[i];
	overlap = calcOverlap(image, Euler, hkls, f_data->nhkls, f_data->nrPxX, f_data->nrPxY,
		recip, outArrThis, f_data->maxNrSpots, rotTranspose, pArr, f_data->pxX, f_data->pxY, f_data->Elo,
		f_data->Ehi);
	return -overlap;
}


inline double FitOrientation(double *image, double euler[3], int *hkls, int nhkls, int nrPxX, int nrPxY,
	double recip[3][3], double *outArrThis, int maxNrSpots, double rotTranspose[3][3], double pArr[3], double pxX, 
	double pxY, double Elo, double Ehi, double tol, double latc[6], double eulerFit[3], double latCUpd[6], double *minVal, int doCrystalFit)
{
	int i,j;
	unsigned n;
	if (doCrystalFit==0){
		n=3;
	} else{
		int non_zero = 0;
		for (i=0;i<6;i++) if (tol_LatC[i]!=0) non_zero++;
		n = 3 + non_zero;
		if (tol_c_over_a !=0){
			n = 4;
		}
	}
	double minf;
	double x[n], xl[n], xu[n];
	x[0] = euler[0]; xl[0] = euler[0] - tol; xu[0] = euler[0]+tol;
	x[1] = euler[1]; xl[1] = euler[1] - tol; xu[1] = euler[1]+tol;
	x[2] = euler[2]; xl[2] = euler[2] - tol; xu[2] = euler[2]+tol;
	if (doCrystalFit !=0){
		int cntr = 3;
		for (i=0;i<6;i++){
			if (tol_LatC[i]!=0){
				x[cntr] = latc[i];
				xl[cntr] = latc[i]*(1-tol_LatC[i]);
				xu[cntr] = latc[i]*(1+tol_LatC[i]);
				cntr++;
			}
		}
		if (tol_c_over_a!=0){
			x[3] = c_over_a_orig;
			xl[3] = c_over_a_orig*(1-tol_c_over_a);
			xu[3] = c_over_a_orig*(1+tol_c_over_a);
		}
	}

	struct dataFit f_data;
	f_data.image = &image[0];
	f_data.hkls = &hkls[0];
	f_data.nhkls = nhkls;
	f_data.nrPxX = nrPxX;
	f_data.nrPxY = nrPxY;
	f_data.outArrThis = &outArrThis[0];
	f_data.maxNrSpots = maxNrSpots;
	for (i=0;i<3;i++){
		f_data.pArr[i] = pArr[i];
		for (j=0;j<3;j++){
			f_data.rotTranspose[i][j] = rotTranspose[i][j];
			f_data.recip[i][j] = recip[i][j];
		}
	}
	for (i=0;i<6;i++) f_data.LatCOrig[i] = latc[i];
	f_data.pxX = pxX;
	f_data.pxY = pxY;
	f_data.Elo = Elo;
	f_data.Ehi = Ehi;
	struct dataFit *f_datat;
	f_datat = &f_data;
	void* trp = (struct dataFit *) f_datat;
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_NELDERMEAD,n);
	nlopt_set_lower_bounds(opt,xl);
	nlopt_set_upper_bounds(opt,xu);
	nlopt_set_min_objective(opt,problem_function,trp);
	nlopt_optimize(opt,x,&minf);
	nlopt_destroy(opt);
	for (i=0;i<3;i++) eulerFit[i] = x[i];
	if (doCrystalFit !=0){
		int cntr2 = 0;
		for (i=0;i<6;i++){
			if (tol_LatC[i]!=0){
				latCUpd[i] = x[3+cntr2];
				cntr2++;
			} else{
				latCUpd[i] = latc[i];
			}
		}
		if (tol_c_over_a!=0){
			double aNew = pow((cellVol/(x[3]*phiVol)),1/3);
			latCUpd[0] = aNew;
			latCUpd[1] = aNew;
			latCUpd[2] = x[3]*aNew;
		}
	}
	*minVal = -minf;
	return 0;
	// *minVal = sqrt(-minf);
}

inline int writeCalcOverlap(double *image, double euler[3], int *hkls, int nhkls, int nrPxX, int nrPxY,
	double recip[3][3], double *outArrThis, int maxNrSpots, double rotTranspose[3][3], double pArr[3], double pxX,
	double pxY, double Elo, double Ehi, FILE *ExtraInfo, int saveExtraInfo, int *simulNrSps){
	int nrSps=0;
	double OM[3][3], OMt[3][3];
	Euler2OrientMat(euler,OMt);
	int i,j;
	double ki[3] = {0,0,1.0};
	MatrixMultF33(OMt,recip,OM);
	int hklnr, badSpot;
	double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3],xp,yp,px,py,sinTheta,E, result=0;
	int spotNr = 0, iterNr, nrPos=0;
	// Save hkl[3], qhat[3], px, py, intensity, spotNr to an array.
	for (hklnr=0;hklnr<nhkls;hklnr++){
		hkl[0] = hkls[hklnr*3+0];
		hkl[1] = hkls[hklnr*3+1];
		hkl[2] = hkls[hklnr*3+2];
		MatrixMultF(OM,hkl,qvec);
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
		px = (xp/pxX) + (0.5*(nrPxX-1));
		if (px <0 || px > (nrPxX-1)) continue;
		py = (yp/pxY) + (0.5*(nrPxY-1));
		if (py <0 || py > (nrPxY-1)) continue;
		sinTheta = -qhat[2];
		E = hc_keVnm * qlen / (4*M_PI*sinTheta);
		if (E < Elo || E > Ehi) continue;
		badSpot = 0;
		for (iterNr=0;iterNr<spotNr;iterNr++){
			if ((fabs(qhat[0] - outArrThis[3*iterNr+0])*100000 < 0.1)&&
			   (fabs(qhat[1] - outArrThis[3*iterNr+1])*100000 < 0.1)&&
			   (fabs(qhat[2] - outArrThis[3*iterNr+2])*100000 < 0.1)) {
					badSpot = 1;
					break;
			   }
		}
		if (badSpot == 0){
			outArrThis[3*spotNr+0] = qhat[0];
			outArrThis[3*spotNr+1] = qhat[1];
			outArrThis[3*spotNr+2] = qhat[2];
			if (image[(int)((int)py*nrPxX+(int)px)] >0){
				// Save hkl[3], qhat[3], px, py, intensity, spotNr to saveArr.
				if (saveExtraInfo !=0){
					#pragma omp critical
					{
						fprintf(ExtraInfo,"%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n", 
							saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1], (int)hkl[2], (int)px, (int)py,
							qhat[0], qhat[1], qhat[2], image[(int)((int)py*nrPxX+(int)px)]);
					}
				}
				result += image[(int)((int)py*nrPxX+(int)px)];
				nrPos++;
			}
			spotNr++;
			if (spotNr == maxNrSpots){
				break;
			}
		}
	}
	*simulNrSps = spotNr;
	nrSps = nrPos;
	result = nrPos*sqrt(result);
	return nrSps;
}


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
	
	if (doFwd==1){
		#pragma omp parallel num_threads(numProcs)
		{
			int procNr = omp_get_thread_num();
			int nrOrientsThread = (int)ceil((double)nrOrients/(double)numProcs);
			uint16_t *outArrThis;
			size_t szArr = nrOrientsThread*(1+2*maxNrSpots);
			size_t OffsetHere;
			OffsetHere = procNr;
			OffsetHere *= szArr;
			OffsetHere *= sizeof(*outArrThis);
			size_t OffsetHereOut;
			OffsetHereOut = procNr;
			OffsetHereOut *= szArr;
			OffsetHereOut *= sizeof(*outArrThis);
			int startOrientNr = procNr * nrOrientsThread;
			int endOrientNr = startOrientNr + nrOrientsThread;
			if (endOrientNr > nrOrients) endOrientNr = nrOrients;
			nrOrientsThread = endOrientNr - startOrientNr;
			szArr = nrOrientsThread*(1+2*maxNrSpots);
			outArrThis = (uint16_t *) calloc(szArr,sizeof(*outArrThis));
			if (outArrThis == NULL){
				printf("Could not allocate outArr per thread, needed %lldMB of RAM. Behavior unexpected now.\n"
					,(long long int)nrOrientsThread*(10+5*maxNrSpots)*sizeof(double)/(1024*1024));
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
		cudaDeviceSynchronize();
		clock_t start = clock();
		cudaMemcpy(device_image,image,nrPxX*nrPxY*sizeof(double),cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		cudaMemset(device_matchedArr,0,nrOrients*sizeof(double));
		cudaDeviceSynchronize();
		// printf("%zu\n",(size_t) (nrOrients+4095)/4096);
		compare<<<(nrOrients+1023)/1024, 1024>>>(nrPxX,nrOrients,maxNrSpots,minIntensity,minNrSpots,device_outArr,device_image,device_matchedArr);
		cudaDeviceSynchronize();
		cudaMemcpy(mArr,device_matchedArr,nrOrients*sizeof(double),cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		clock_t end = clock();
		printf("Time elapsed to match on the GPU: %lf\n",(double)(end-start)/CLOCKS_PER_SEC);
		size_t nrMatches=0;
		for (int i=0;i<nrOrients;i++){
			if (mArr[i]>0){
				matchedArr[i] = mArr[i];
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