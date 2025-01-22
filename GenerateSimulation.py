#!/usr/bin/env python

import numpy as np
import h5py
from math import pi, sin, cos
import scipy.ndimage as ndimg
from PIL import Image
import argparse
import os, sys
import subprocess
pytpath = sys.executable
installPath = os.path.dirname(os.path.abspath(__file__))

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

#### ONLY IMPLEMENTED TO USE ASTAR (CUBIC)

parser = MyParser(description='''LaueMatching Generate Simulation using an experiment configuration and a list of orientations, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-configFile', type=str, required=True, help='Configuration file to run the simulation.')
parser.add_argument('-orientationFile', type=str, required=True, help='File containing list of orientations to simulate. Must be orientation matrices with determinant 1. Each row has one orientation.')
parser.add_argument('-outputFile', type=str, required=True, help='File to save the data. Must be of the format FILESTEM_XX.h5, where XX is the file number, eg. 1.')
args, unparsed = parser.parse_known_args()
configFile = args.configFile
orientationFN = args.orientationFile
outFN = args.outputFile

### Get parameters
hklf = 'valid_reflections.csv'
sym= 'F'
lines = open(configFile,'r').readlines()
astar = -1
for line in lines:
	if line.startswith('SpaceGroup'):
		sgNum = int(line.split()[1])
	elif line.startswith('Symmetry'):
		sym = line.split()[1]
		if sym not in 'FICAR' and len(sym) != 1:
			print('Invalid value for sym, must be one character from F,I,C,A,R')
			sys.exit()
	elif line.startswith('LatticeParameter'):
		latC = ' '.join(line.split()[1:7])
	elif line.startswith('R_Array'):
		r_arr = ' '.join(line.split()[1:4])
	elif line.startswith('P_Array'):
		p_arr = ' '.join(line.split()[1:4])
	elif line.startswith('PxX'):
		dx = float(line.split()[1])
	elif line.startswith('PxY'):
		dy = float(line.split()[1])
	elif line.startswith('AStar'):
		astar = float(line.split()[1])
	elif line.startswith('Elo'):
		Elo = float(line.split()[1])
	elif line.startswith('Ehi'):
		Ehi = float(line.split()[1])
	elif line.startswith('SimulationSmoothingWidth'):
		gaussWidth = int(line.split()[1])
	elif line.startswith('NrPxX'):
		nPxX = int(line.split()[1])
	elif line.startswith('NrPxY'):
		nPxY = int(line.split()[1])
	elif line.startswith('HKLFile'):
		hklf = line.split()[1]

P = np.array([float(p) for p in p_arr.split()])
R = np.array([float(r) for r in r_arr.split()])
Nx = nPxX
Ny = nPxY
if astar == -1:
	astar = 2*pi/float(latC[0])

lines = open(orientationFN).readlines()
if lines[0].startswith('%GrainNr'):
	orientations = np.genfromtxt(orientationFN,skip_header=1)[:,22:31]
else:
	orientations = np.genfromtxt(orientationFN)

# If hkl file does not exist, generate.
if not os.path.exists(hklf):
	cmmd = f'{pytpath} {installPath}/GenerateHKLs.py -resultFileName {hklf} -sgnum {sgNum} -sym {sym} '
	cmmd += f'-latticeParameter {latC} -RArray {r_arr} -PArray {p_arr} -NumPxX {nPxX} -NumPxY {nPxY} '
	cmmd += f'-dx {dx} -dy {dy}'
	subprocess.call(cmmd,shell=True)

hklarr = np.genfromtxt(hklf)


hc_keVnm = 1.2398419739

def XYZ2pixel(XYZ):
	xyz = np.linalg.inv(rot).dot(XYZ.T).T
	z = xyz.item(0,2)
	if z <= 0: return None
	scale = P[2] / z
	xyz = xyz * scale
	xp = xyz.item(0,0) - P[0]
	yp = xyz.item(0,1) - P[1]
	px = xp / dx + 0.5*(Nx-1)
	py = yp / dy + 0.5*(Ny-1)
	if px < 0 or px >= (Nx-1): return None
	if py < 0 or py >= (Ny-1): return None
	return (px, py)

def getSpots(recip,nr):
	global posArr
	hkls = np.copy(hklarr[:,:3])
	qvecs = recip.dot(hkls.T).T
	qlens = np.linalg.norm(qvecs,axis=1)
	good = qlens > 0
	qvecs = qvecs[good,:]
	qlens = qlens[good]
	qhats = np.divide(qvecs,np.matrix([qlens,qlens,qlens]).T)
	dots = np.squeeze(np.copy(qhats[:,2]))
	mdots = np.empty(qhats.shape)
	mdots[:,0] = np.copy(dots)
	mdots[:,1] = np.copy(dots)
	mdots[:,2] = np.copy(dots)
	kfs = ki- 2*np.multiply(mdots,qhats)
	xyz = np.dot(np.linalg.inv(rot),kfs.T).T
	z = np.squeeze(np.copy(xyz[:,2]))
	goodZ = z > 0
	qlens = qlens[goodZ]
	qhats = qhats[goodZ,:]
	xyz = xyz[goodZ,:]
	xyz = np.divide(xyz * P[2],xyz[:,2])
	xp = xyz[:,0] - P[0]
	yp = xyz[:,1] - P[1]
	px = xp / dx + 0.5*(Nx-1)
	py = yp / dy + 0.5*(Ny-1)
	px = np.squeeze(px)
	py = np.squeeze(py)
	good = (px >= 0) & (px < (Nx-1)) & (py >= 0) & (py < (Ny-1))
	px = px[good]
	py = py[good]
	pixels = np.vstack((px,py)).T
	qlens = qlens[good.flat]
	qhats = qhats[good.flat,:]
	sinThetas = -qhats[:,2]
	Es = hc_keVnm*np.divide(qlens,sinThetas.T).T/(4*pi)
	goodE = (Es > Elo) & (Es < Ehi)
	pixels = pixels[goodE.flat,:]
	nrPx = 0
	for pixel in pixels:
		posArr.append([pixel.item(1),pixel.item(0),nr])
		if np.random.randint(0,10) > 2:
			nrPx+=1
			img[int(pixel.item(1)),int(pixel.item(0))] = np.random.randint(500,16000)
			posArr[-1].append(1)
		else:
			posArr[-1].append(0)
	print(f'Number of spots: {nrPx}')

img = np.zeros((Nx,Ny))
posArr = []

rotang = np.linalg.norm(R)
rotvect = R/np.linalg.norm(R)
rot = np.matrix([[cos(rotang)+(1-cos(rotang))*(rotvect[0]**2),                  (1-cos(rotang))*rotvect[0]*rotvect[1]-sin(rotang)*rotvect[2], (1-cos(rotang))*rotvect[0]*rotvect[2]+sin(rotang)*rotvect[1]],
                 [(1-cos(rotang))*rotvect[1]*rotvect[0]+sin(rotang)*rotvect[2], cos(rotang)+(1-cos(rotang))*(rotvect[1]**2),                  (1-cos(rotang))*rotvect[1]*rotvect[2]-sin(rotang)*rotvect[0]],
                 [(1-cos(rotang))*rotvect[2]*rotvect[0]-sin(rotang)*rotvect[1], (1-cos(rotang))*rotvect[2]*rotvect[1]+sin(rotang)*rotvect[0], cos(rotang)+(1-cos(rotang))*(rotvect[2]**2)]])
ki = np.matrix([0,0,1.0])
recips = orientations*astar

nr = 0
for recip in recips:
	recip = recip.reshape((3,3))
	getSpots(recip,nr)
	nr+=1

img = ndimg.gaussian_filter(img,gaussWidth).astype(np.uint16)
Image.fromarray(img).save(outFN+'.tif')
hFile = h5py.File(outFN,'w')
hFile.create_dataset('/entry1/data/data',data=img)
hFile.create_dataset('/entry1/recips',data=recips)
posArr = np.array(posArr)
hFile.create_dataset('/entry1/spots',data=posArr)
hFile.create_dataset('/entry1/orientation_matrices',data=orientations)
hFile.close()