import sys
import math
import numpy as np
from math import cos,sin,sqrt
import argparse
import warnings

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

warnings.filterwarnings('ignore')
parser = MyParser(description='''
LaueMatching compute valid reflections.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFileName', type=str, required=False, default='valid_reflections.csv', help='Name of output file')
parser.add_argument('-sym', type=str, required=False, default='F', help='First character of the Hermann-Maguin description. Accpted values: F,I,C,A,R')
parser.add_argument('-sgnum', type=int, required=False, default=225, help='SpaceGroup number. Between 1-230.')
parser.add_argument('-latticeParameter', type=float, nargs=6, required=True, help='Lattice Parameter a,b,c,alpha,beta,gamma [nm, nm, nm, degrees, degrees, degrees].')
parser.add_argument('-RArray', type=float, nargs=3, required=True, help='Rotation array describing detector orientation. 3 values.[degrees]')
parser.add_argument('-PArray', type=float, nargs=3, required=True, help='Translation array describing detector position. 3 values.[m]')
parser.add_argument('-NumPxX', type=float, required=False, default=2048, help='Number of pixels in X direction.')
parser.add_argument('-NumPxY', type=float, required=False, default=2048, help='Number of pixels in Y direction.')
parser.add_argument('-dx', type=float, required=False, default=200e-6, help='Pixel size in X direction.[m]')
parser.add_argument('-dy', type=float, required=False, default=200e-6, help='Pixel size in Y direction.[m]')
args, unparsed = parser.parse_known_args()
fn = args.resultFileName
sym = args.sym
if sym not in 'FICAR' and len(sym) != 1:
    print('Invalid value for sym, must be one character from F,I,C,A,R')
    sys.exit()
LatC = args.latticeParameter
sgNum = args.sgnum
R_ = np.asarray(args.RArray)
P_ = np.asarray(args.PArray)
Nx = args.NumPxX
Ny = args.NumPxY
dx = args.dx
dy = args.dy

hc_keVnm = 1.2398419739
rad2deg = 180/np.pi
deg2rad = np.pi/180

def zeroOut(val):
	if np.abs(val) < 0.00000000001:
		return 0
	else:
		return val

def calcRecipArray(Lat, SpaceGroup):
	recip = np.zeros((3,3))
	a = Lat[0]; b = Lat[1]; c= Lat[2]; alpha = Lat[3]; beta = Lat[4]; gamma = Lat[5];
	rhomb = 0
	if (SpaceGroup == 146 or SpaceGroup == 148 or SpaceGroup == 155 or SpaceGroup == 160 or
		SpaceGroup == 161 or SpaceGroup == 166 or SpaceGroup == 167):
		rhomb = 1
	ca = cos((alpha)* deg2rad)
	cb = cos((beta)* deg2rad)
	cg = cos((gamma)* deg2rad)
	sg = sin((gamma)* deg2rad)
	phi = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
	Vc = a*b*c * phi
	pv = (2*np.pi) / (Vc)
	if (rhomb == 0):
		a0 = a			; a1 = 0.0				; a2 = 0.0;
		b0 = b*cg		; b1 = b*sg				; b2=0;
		c0 = c*cb		; c1 = c*(ca-cb*cg)/sg	; c2=c*phi/sg;
		a0 = zeroOut(a0); a1 = zeroOut(a1); a2 = zeroOut(a2);
		b1 = zeroOut(b1); b2 = zeroOut(b2);
		c2 = zeroOut(c2);
	else:
		p = sqrt(1.0 + 2*ca);
		q = sqrt(1.0 - ca);
		pmq = (a/3.0)*(p-q);
		p2q = (a/3.0)*(p+2*q);
		a0 = p2q; a1 = pmq; a2 = pmq;
		b0 = pmq; b1 = p2q; b2 = pmq;
		c0 = pmq; c1 = pmq; c2 = p2q;
	recip[0][0] = zeroOut((b1*c2-b2*c1)*pv);
	recip[1][0] = zeroOut((b2*c0-b0*c2)*pv);
	recip[2][0] = zeroOut((b0*c1-b1*c0)*pv);
	recip[0][1] = zeroOut((c1*a2-c2*a1)*pv);
	recip[1][1] = zeroOut((c2*a0-c0*a2)*pv);
	recip[2][1] = zeroOut((c0*a1-c1*a0)*pv);
	recip[0][2] = zeroOut((a1*b2-a2*b1)*pv);
	recip[1][2] = zeroOut((a2*b0-a0*b2)*pv);
	recip[2][2] = zeroOut((a0*b1-a1*b0)*pv);
	return recip

def _ALLOW_FC(h,k,l):
    """ face-centered, hkl must be all even or all odd """
    return (h+k) % 2 == 0 and (k+l) % 2 == 0

def _ALLOW_BC(h,k,l):
    """ body-centered, !mod(round(h+k+l),2), sum must be even """
    return (h+k+l) % 2 == 0

def _ALLOW_CC(h,k,l):
    """ C-centered, !mod(round(h+k),2) """
    return ((h+k) % 2)==0

def _ALLOW_AC(h,k,l):
    """ A-centered, !mod(round(k+l),2) """
    return ((k+l) % 2)==0

def _ALLOW_RHOM_HEX(h,k,l):
    """ rhombohedral hexagonal, allowed are -H+K+L=3n or H-K+L=3n """
    return (-h+k+l)%3 == 0 or (h-k+l)%3 == 0

def _ALLOW_HEXAGONAL(h,k,l):
    """ hexagonal, forbidden are: H+2K=3N with L odd """
    return bool((h+2*k)%3) or not bool(l%2)

class detectorType(object):

    def __init__(self, Nx, Ny, dx, dy, R, P, name=''):
        try:
            if Nx <= 2: raise ValueError('Nx = ' + repr(Nx))
            if Ny <= 2: raise ValueError('Ny = ' + repr(Ny))
            Nx = round(Nx)
            Ny = round(Ny)

            if dx <= 0: raise ValueError('dx = ' + repr(dx))
            if dy <= 0: raise ValueError('dy = ' + repr(dy))

        except:
            return None

        self.name = name
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.P = np.asarray([P[0],P[1],P[2]])
        rotang = np.linalg.norm(R)
        rotvect = R/np.linalg.norm(R);
        self.rot = np.matrix([[math.cos(rotang)+(1-math.cos(rotang))*(rotvect[0]**2), (1-math.cos(rotang))*rotvect[0]*rotvect[1]-math.sin(rotang)*rotvect[2], (1-math.cos(rotang))*rotvect[0]*rotvect[2]+math.sin(rotang)*rotvect[1]],
                              [(1-math.cos(rotang))*rotvect[1]*rotvect[0]+math.sin(rotang)*rotvect[2], math.cos(rotang)+(1-math.cos(rotang))*(rotvect[1]**2),  (1-math.cos(rotang))*rotvect[1]*rotvect[2]-math.sin(rotang)*rotvect[0]],
                              [(1-math.cos(rotang))*rotvect[2]*rotvect[0]-math.sin(rotang)*rotvect[1], (1-math.cos(rotang))*rotvect[2]*rotvect[1]+math.sin(rotang)*rotvect[0], math.cos(rotang)+(1-math.cos(rotang))*(rotvect[2]**2)]
                              ]);
        self.pCenter = ((self.Nx-1)/2, (self.Ny-1)/2)	# center of detector (pixels)
        self.XYZcenter = self.pixel2XYZ(self.pCenter[0],self.pCenter[1])	# center of detector in beam-line corrdinates

    def pixel2XYZ(self, px, py):					# given (px,py), compute beam-line coords XYZ
        try:
            px = np.double(px)
            py = np.double(py)
        except:
            ValueError('cannot interpret pixel:' + repr(px) + '  ' + repr(py))
        if px < 0 or px >= self.Nx: return None		# must lie on detector
        if py < 0 or py >= self.Ny: return None

        # x' and y' (requiring z'=0), detector starts centered on origin and perpendicular to z-axis
        xp = (px - 0.5*(self.Nx-1)) * self.dx # (x' y' z'), position on detector
        yp = (py - 0.5*(self.Ny-1)) * self.dy
        xp += self.P[0]# translate by P
        yp += self.P[1]
        zp =  self.P[2]
        xyz = np.matrix([xp,yp,zp])# position in detector frame
        XYZ = self.rot.dot(xyz.T).flatten()
        return XYZ

    def XYZ2pixel(self, XYZ):
        try:
            if XYZ.shape != (1,3): return None
        except: return None
        xyz = np.linalg.inv(self.rot).dot(XYZ.T).T	# un-rotate

        z = xyz.item(0,2)
        if z <= 0: return None						# does not point to detector
        scale = self.P[2] / z						# scale xyz so that z = dist
        xyz = xyz * scale

        xp = xyz.item(0,0) - self.P[0]				# un-translate by P[]
        yp = xyz.item(0,1) - self.P[1]
        px = xp / self.dx + 0.5*(self.Nx-1) 		# (x' y' z') position on detector --> pixel
        py = yp / self.dy + 0.5*(self.Ny-1)

        if px < 0 or px >= (self.Nx-1): return None	# must land on detector
        if py < 0 or py >= (self.Ny-1): return None

        return (px, py)

def preCalcHKLs(detector,LatC,recip,Ehi=30):
    XYZ = detector.pixel2XYZ((detector.Nx-1)/2, detector.Ny-1)
    thMax = math.atan2(XYZ.item((0,1)), XYZ.item((0,2))) / 2
    Qmax = 4*math.pi * math.sin(thMax) * Ehi/hc_keVnm
    hmax = int( math.floor( Qmax / (2*math.pi / LatC[0])) )
    kmax = int( math.floor( Qmax / (2*math.pi / LatC[1])) )
    lmax = int( math.floor( Qmax / (2*math.pi / LatC[2])) )
    goodArr = np.zeros((hmax*2+1,kmax*2+1,lmax*2+1))
    for l in range(-lmax,lmax+1):
        for k in range(-kmax,kmax+1):
            for h in range(-hmax,hmax+1):
                if h==0 and k==0 and l==0: continue
                # hkl = np.matrix([h,k,l])
                # qvec = recip.dot(hkl.T).T
                # qlens = np.linalg.norm(qvec)
                # if qlens == 0: 
                #     continue
                # qhat = qvec / qlens
                # sinTheta = -qhat.item((0,2))
                # if sinTheta == 0:
                #     continue
                if sym == 'F':
                    if not _ALLOW_FC(h,k,l): continue
                elif sym == 'I':
                    if not _ALLOW_BC(h,k,l): continue
                elif sym == 'C':
                    if not _ALLOW_CC(h,k,l): continue
                elif sym == 'A':
                    if not _ALLOW_AC(h,k,l): continue
                elif sym == 'R' and usingHexAxes:
                    if not _ALLOW_RHOM_HEX(h,k,l): continue # rhombohedral cell with hexagonal axes
                if Hexagonal:			# Hexagonal
                    if not _ALLOW_HEXAGONAL(h,k,l): continue
                goodArr[hmax-h,kmax-k,lmax-l] = 1
    hklarr = np.zeros((int(np.sum(goodArr)),4))
    ctr = 0
    for l in range(-lmax,lmax+1):
        for k in range(-kmax,kmax+1):
            for h in range(-hmax,hmax+1):
                if goodArr[hmax-h,kmax-k,lmax-l] == 1:
                    hklarr[ctr,0] = h
                    hklarr[ctr,1] = k
                    hklarr[ctr,2] = l
                    hklarr[ctr,3] = h*h + k*k + l*l
                    ctr += 1
    return hklarr

usingHexAxes = (abs(90-LatC[3])+abs(90-LatC[4])+abs(120-LatC[5]))<1e-6
if sgNum >=168 and sgNum<195:
    Hexagonal = 1
else:
    Hexagonal = 0
detector = detectorType(Nx=Nx, Ny=Ny, dx=dx, dy=dy, R=R_, P=P_, name='Perkn-Elmer')

recip = calcRecipArray(LatC,sgNum)
om = np.array([[0.8340462 , -0.54723977 , 0.06996829 ],[ 0.51044064 , 0.71732763 ,-0.47422719 ],[ 0.20932579 , 0.43124205 , 0.87761781]]) # Random orientation
recip = recip.dot(om)
hklarr = preCalcHKLs(detector,LatC,recip=recip)
print(f'Number of valid HKLs: {hklarr.shape[0]}')
hklarr2 = hklarr[hklarr[:,3].argsort()]
np.savetxt(fn,hklarr2,fmt="%d %d %d %d")
