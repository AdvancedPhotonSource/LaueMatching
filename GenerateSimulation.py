import numpy as np
import h5py
from math import pi, sin, cos
import scipy.ndimage as ndimg
from PIL import Image

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

def getSpots(recip):
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
		if np.random.randint(0,10) > 3:
			nrPx+=1
			img[int(pixel.item(1)),int(pixel.item(0))] = np.random.randint(0,16000)
	print(nrPx)


# Read grain orientations, hkls
orientations = np.genfromtxt('Grains.csv',skip_header=9)[:,1:10]
hklarr = np.genfromtxt('valid_reflections.csv')

astar = 17.8307091979
P = np.array([0.028745, 0.002788, 0.513115])
R = np.array([-1.20131258, -1.21399082, -1.21881158])
dx = 0.0002
dy = 0.0002
Nx = 2048
Ny = 2048
Elo = 5
Ehi = 30
gaussWidth = 2

img = np.zeros((Nx,Ny))

rotang = np.linalg.norm(R)
rotvect = R/np.linalg.norm(R)
rot = np.matrix([[cos(rotang)+(1-cos(rotang))*(rotvect[0]**2), (1-cos(rotang))*rotvect[0]*rotvect[1]-sin(rotang)*rotvect[2], (1-cos(rotang))*rotvect[0]*rotvect[2]+sin(rotang)*rotvect[1]],
                              [(1-cos(rotang))*rotvect[1]*rotvect[0]+sin(rotang)*rotvect[2], cos(rotang)+(1-cos(rotang))*(rotvect[1]**2),  (1-cos(rotang))*rotvect[1]*rotvect[2]-sin(rotang)*rotvect[0]],
                              [(1-cos(rotang))*rotvect[2]*rotvect[0]-sin(rotang)*rotvect[1], (1-cos(rotang))*rotvect[2]*rotvect[1]+sin(rotang)*rotvect[0], cos(rotang)+(1-cos(rotang))*(rotvect[2]**2)]
                              ]);
ki = np.matrix([0,0,1.0])
recips = orientations*astar

for recip in recips:
	recip = recip.reshape((3,3))
	getSpots(recip)

img = ndimg.gaussian_filter(img,gaussWidth).astype(np.uint16)
Image.fromarray(img).save('simulated_data.tif')
hFile = h5py.File('simulated_data.h5','w')
hFile.create_dataset('/entry1/data/data',data=img)
np.savetxt('simulated_recips.txt',recips,fmt='%.6f',delimiter=' ')
