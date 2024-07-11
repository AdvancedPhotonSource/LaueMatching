#!/usr/bin/env python

import h5py
import numpy as np
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
import math
import os, sys
import subprocess
import shutil
from PIL import Image
import diplib as dip
import skimage
import argparse
plt.rcParams['font.size'] = 3
pytpath = sys.executable
installPath = os.path.dirname(os.path.abspath(__file__))

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''LaueMatching Index Images, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-configFile', type=str, required=True, help='Configuration file to run the analysis.')
parser.add_argument('-imageFile', type=str, required=True, help='Image FileName, if you want multiple files, give first filename.')
parser.add_argument('-nCPUs', type=int, required=True, help='Number of CPU cores to use. Both GPU and CPU codes use this.')
parser.add_argument('-computeType', type=str, required=False, default='CPU', help='Computation type: provide either CPU or GPU.')
parser.add_argument('-nFiles', type=int, required=False, default=1, help='Number of files to run. Files must be of the format: FILESTEM_XX.h5, where XX is the filenumber.')
args, unparsed = parser.parse_known_args()
configFile = args.configFile
imageFile = args.imageFile
nCPUs = args.nCPUs
computeType = args.computeType
nFiles = args.nFiles

if computeType not in ["CPU","GPU"]:
	print("Compute type MUST be either CPU or GPU. Please provide either.")

if nFiles == 1:
	imageFiles = [imageFile]
else:
	fstem = '_'.join(imageFile.split('_')[:-1])
	last = imageFile.split('_')[-1]
	ext = f'.{".".join(last.split(".")[1:])}'
	stNr = int(last.split(ext)[0])
	imageFiles = []
	for fNr in range(stNr,stNr+nFiles):
		imageFiles.append(f'{fstem}_{fNr}{ext}')
	print("Files to be processed: ",imageFiles)

# sys.exit()

lines = open(configFile,'r').readlines()
nPasses = 5
filtRad = 101
minArea = 10
backgroundFN = 'median.bin'
sym = 'F'
thresh = 0
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
		distt = float(line.split()[3])
	elif line.startswith('Threshold'):
		thresh = float(line.split()[1])
	elif line.startswith('MinIntensity'):
		minIntensity = float(line.split()[1])
	elif line.startswith('PxX'):
		px = float(line.split()[1])
		dx = px
	elif line.startswith('PxY'):
		dy = float(line.split()[1])
	elif line.startswith('OrientationSpacing'):
		orientSpacing = float(line.split()[1])
	elif line.startswith('NrPxX'):
		nrPixels = int(line.split()[1])
		nPxX = nrPixels
	elif line.startswith('NrPxY'):
		nPxY = int(line.split()[1])
	elif line.startswith('FilterRadius'):
		filtRad = int(line.split()[1])
	elif line.startswith('NMeadianPasses'):
		nPasses = int(line.split()[1])
	elif line.startswith('MinArea'):
		minArea = int(line.split()[1])
	elif line.startswith('MinGoodSpots'):
		minGoodSpots = int(line.split()[1])
	elif line.startswith('MaxNrLaueSpots'):
		maxLaueSpots = int(line.split()[1])
	elif line.startswith('ResultDir'):
		resultdir = line.split()[1]
	elif line.startswith('OrientationFile'):
		orientf = line.split()[1]
	elif line.startswith('HKLFile'):
		hklf = line.split()[1]
	elif line.startswith('BackgroundFile'):
		backgroundFN = line.split()[1]
	elif line.startswith('ForwardFile'):
		fwdf = line.split()[1]

outdpi = 600
scalarX = nrPixels/outdpi
scalarY = nrPixels/outdpi
plt.rcParams['figure.figsize'] = [scalarX, scalarY]

# If orientf does not exist, copy it from the repo directory
if not os.path.exists(orientf):
	shutil.copy2(f'{installPath}/100MilOrients.bin',orientf)

# Check if the correct fwd simulation exists based on size of the file only, otherwise enable simulation.
if os.path.exists(fwdf):
	size_orient = os.path.getsize(orientf)
	nrOrient = int(size_orient/(8*9))
	size_fwd = os.path.getsize(fwdf)
	nrOrientFwd = int(size_fwd/(2*(1+2*maxLaueSpots)))
	if nrOrient != nrOrientFwd:
		doFwd = 'DoFwd 1\n'
		os.remove(fwdf)
	else: 
		doFwd = 'DoFwd 0\n'
else:
	doFwd = 'DoFwd 1\n'

# If hkl file does not exist, generate.
if not os.path.exists(hklf):
	cmmd = f'{pytpath} {installPath}/GenerateHKLs.py -resultFileName {hklf} -sgnum {sgNum} -sym {sym} '
	cmmd += f'-latticeParameter {latC} -RArray {r_arr} -PArray {p_arr} -NumPxX {nPxX} -NumPxY {nPxY} '
	cmmd += f'-dx {dx} -dy {dy}'
	subprocess.call(cmmd,shell=True)

if os.path.exists(resultdir):
	shutil.rmtree(resultdir,ignore_errors=True)
os.makedirs(resultdir,exist_ok=True)

print("Running Laue Matching")

file_path = os.path.dirname(os.path.realpath(__file__))
env = dict(os.environ)
libpth = os.environ.get('LD_LIBRARY_PATH','')

def indices_to_text(h,k,l):
	t = r""
	for index in (h, k, l):
		if index < 0:
			t = t + r"$\overline{" + f"{abs(index)}" + r"}$ "
		else:
			t = t + f"{index} "
	return t + ""

def calcL2(l1,l2):
	return math.sqrt((l1[0]-l2[0])**2+(l1[1]-l2[1])**2)

def runFile(imageFN):
	global thresh
	h_image = h5py.File(imageFN,'r')
	h_im = np.array(h_image['/entry1/data/data'][()])
	h_im_raw = np.copy(h_im)
	
	# Check if background exists:
	if not os.path.exists(backgroundFN):
		background = dip.Image(h_im)
		for i in range(nPasses):
			background = dip.MedianFilter(background,filtRad)
		np.array(background).astype(np.double).tofile(backgroundFN)
	else:
		background = np.fromfile(backgroundFN,dtype=np.double).reshape((nPxX,nPxY))
	
	h_im_corr = h_im.astype(np.double) - background
	threshT = 60 * (1+np.std(h_im_corr)//60)
	if thresh > threshT:
		threshT = thresh # Use larger value
	print(f'Computed/input threshold: {threshT}')
	h_im_corr[h_im_corr < threshT] = 0
	h_im = h_im_corr.astype(np.uint16)

	imageFN = resultdir + '/'+ imageFN
	hf_out = h5py.File(imageFN+'.bin.output.h5','w')
	print(f'Processing file {imageFN}.bin')
	hf_out.create_dataset('/entry/data/raw_data',data=h_im_raw)
	hf_out.create_dataset('/entry/data/cleaned_data_threshold',data=h_im)

	# Do connected components and filter smaller spots
	labels,nlabels = ndimg.label(h_im)
	hf_out.create_dataset('/entry/data/cleaned_data_threshold_labels_unfiltered',data=labels)
	centers = []
	for lNr in range(1,nlabels):
		if (np.sum(labels==lNr) > minArea):
			com = ndimg.center_of_mass(h_im,labels,lNr)
			centers.append([lNr,com,np.sum(labels==lNr)])
		else:
			h_im[labels==lNr] = 0
			labels[labels==lNr] = 0
	hf_out.create_dataset('/entry/data/cleaned_data_threshold_filtered',data=h_im)
	hf_out.create_dataset('/entry/data/cleaned_data_threshold_filtered_labels',data=labels)
	# Image.fromarray(h_im).save(imageFN+'.bin.input.tif')
	
	# Find the Gaussian Width to use for blurring
	bestLen = nrPixels*2
	for l1 in range(len(centers)-1):
		cent1 = centers[l1][1]
		bestLenThis = nrPixels*2
		for l2 in range(l1+1,len(centers)):
			cent2 = centers[l2][1]
			l2_norm = calcL2(cent1,cent2)
			if l2_norm < bestLenThis:
				bestLenThis = l2_norm
		if (bestLenThis < bestLen): bestLen = bestLenThis
	deltaPos = math.tan(orientSpacing/(2*180/math.pi))*distt/px
	# We have deltaPos and bestLenThis, we should use gaussWidth
	gaussWidth = int(math.ceil(0.25*math.ceil(min(deltaPos,bestLenThis))))
	h_im2 = ndimg.gaussian_filter(h_im,gaussWidth)

	# labels = skimage.segmentation.watershed(-h_im,mask=h_im,connectivity=2)
	# nlabels = np.max(labels)

	# Use watershed to find the labels
	labels2 = skimage.segmentation.watershed(-h_im2,mask=h_im2,connectivity=2)
	nlabels2 = np.max(labels2)

	# Hard coded that maximum number of orientations can be nSpots/5.
	colors = plt.get_cmap('nipy_spectral',nlabels2/5)

	h_im2.astype(np.double).tofile(imageFN+'.bin') # THIS IS THE FILE FED TO THE INDEXING C/CU CODE.
	hf_out.create_dataset('/entry/data/input_blurred',data=h_im2)
	# Image.fromarray(h_im2).save(imageFN+'.bin.inputBlurred.tif')

	### RUN INDEXING
	fout = open(imageFN+'.LaueMatching_stdout.txt','w')
	if computeType in "CPU":
		cmmd = f'{file_path}/bin/LaueMatchingCPU {configFile} {orientf} {hklf} {imageFN}.bin {nCPUs}'
	else:
		cmmd = f'{file_path}/bin/LaueMatchingGPU {configFile} {orientf} {hklf} {imageFN}.bin {nCPUs}'
	print(f'Command to be run: {cmmd}')
	env['LD_LIBRARY_PATH'] = f'{file_path}/LIBS/NLOPT/lib:{file_path}/LIBS/NLOPT/lib64:{libpth}'
	subprocess.call(cmmd,shell=True,env=env,stdout=fout)
	fout.close()

	# Read solutions and spots
	orientationInfo = np.genfromtxt(imageFN+'.bin.solutions.txt',skip_header=1)
	headgr = open(imageFN+'.bin.solutions.txt').readline()
	spotInfo = np.genfromtxt(imageFN+'.bin.spots.txt',skip_header=1)
	headsp = open(imageFN+'.bin.spots.txt').readline()

	# Write some stuff
	or_gr = hf_out.create_dataset('/entry/results/original_orientations',data=orientationInfo)
	or_gr.attrs['head'] = np.string_(headgr)
	or_sp = hf_out.create_dataset('/entry/results/original_spots',data=spotInfo)
	or_sp.attrs['head'] = np.string_(headsp)
	outfor = open(imageFN+'.bin.good_solutions.txt','a')
	outfsp = open(imageFN+'.bin.good_spots.txt','a')
	outfor.write(headgr)
	outfsp.write(headsp)

	# orientations = []
	# spots = []
	# legends=[]

	nsols = 0
	label_found = []
	fmtout = '%4d\t%4d\t%11.2f\t%11.2f\t%12.3f\t%4d\t%4d\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t'
	fmtout += '%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f'
	fmtout += '\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%12.6f\t%9.6f\t%10d'

	if (len(orientationInfo.shape)==2): orientationInfo = orientationInfo[np.argsort(-orientationInfo[:,4])]
	else: orientationInfo = np.expand_dims(orientationInfo,axis=0)

	# Make a figure with the image on the background
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig,[0.,0.,1.,1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	# fig,ax = plt.subplots()
	h_im_plt = h_im
	h_im_plt[h_im_plt==0] = 1
	ax.imshow(np.log(h_im_plt),cmap='Greens')

	orientationNr = 0
	for orientation in orientationInfo:
		spotThis = spotInfo[spotInfo[:,0]==orientation[0],:]
		goodSpots = spotThis[spotThis[:,-1]>=threshT/2,:] # This will ensure that we had intensity greater than the original threshold/2 to account for blurring
		if goodSpots.shape[0] >= minGoodSpots:
			# Find which labels were already found, remove those spots from list
			badRows = []
			for spotNr in range(goodSpots.shape[0]):
				spot = goodSpots[spotNr]
				if labels2[int(spot[6])][int(spot[5])]:
					if labels2[int(spot[6])][int(spot[5])] in label_found:
						# Remove this spot from the list
						badRows.append(spotNr)
			badRows.reverse()
			if goodSpots.shape[0] - len(badRows) < minGoodSpots:
				continue
			for rowNr in badRows:
				goodSpots = np.delete(goodSpots,rowNr,0)
			#Figure out which labels were found
			for spot in goodSpots:
				indices = indices_to_text(int(spot[2]),int(spot[3]),int(spot[4]))
				ax.text(spot[5]-np.random.randint(0,20),spot[6]-20,indices,c=colors(orientationNr))
				if labels2[int(spot[6])][int(spot[5])]:
					label_found.append(labels2[int(spot[6])][int(spot[5])])
			# We need to check for the spots that were overlapping, but in the gaussian blurred image
			orientation[5] = goodSpots.shape[0]
			thisID = orientation[0]
			nsols+=1
			orientation = orientation.reshape(orientation.shape[0],-1).transpose()
			np.savetxt(outfor,orientation,fmt=fmtout)
			np.savetxt(outfsp,goodSpots,fmt='%4d\t%3d\t%3d\t%3d\t%3d\t%5d\t%5d\t%9.6f\t%9.6f\t%9.6f\t%7d')
			# Save an image with the blobs from found spots as open squares and orientation id
			lbl = 'OrientationID '+str(int(thisID))
			ax.plot(goodSpots[:,5],goodSpots[:,6],'ks', markerfacecolor='none', ms=3, markeredgecolor=colors(orientationNr),markeredgewidth=0.1,label=lbl)
			orientationNr+=1
	plt.legend()
	plt.savefig(imageFN+'.bin.LabeledImage.tif',dpi=outdpi)
	l_im = np.array(Image.open(imageFN+'.bin.LabeledImage.tif').convert('RGB'))
	hf_out.create_dataset('/entry/results/LabeledImage',data=l_im)
	outfor.close()
	outfsp.close()
	goodGrs = np.genfromtxt(imageFN+'.bin.good_solutions.txt',skip_header=1)
	goodSps = np.genfromtxt(imageFN+'.bin.good_spots.txt',skip_header=1)
	go_gr = hf_out.create_dataset('/entry/results/filtered_orientations',data=goodGrs)
	go_gr.attrs['head'] = np.string_(headgr)
	go_sp = hf_out.create_dataset('/entry/results/filtered_spots',data=goodSps)
	go_sp.attrs['head'] = np.string_(headsp)
	# Find the labels found, remove them and save a figure with those spots only.
	label_found = set(label_found)
	print(f'Blurring width of {gaussWidth} pixels was used.')
	print(f'{len(centers)} peaks in original image, {len(label_found)} successfully identified.')
	print(f'{nsols} orientations were found after filtering out intensity and duplicate matches.')
	im_notfound = np.copy(h_im)
	im_found = np.zeros((h_im.shape))
	for label in label_found:
		im_found[labels2==label] = 255
		im_notfound[labels2==label] = 0
	hf_out.create_dataset('/entry/results/found_pixels',data=im_found)
	hf_out.create_dataset('/entry/results/not_found_pixels',data=im_notfound)
	hf_out.create_dataset('/entry/results/not_found_pixels_blurred',data=ndimg.gaussian_filter(im_notfound,gaussWidth))
	hf_out.close()

	# Image.fromarray(im_found).save(imageFN+'.bin.FoundPixels.tif')
	# Image.fromarray(im_notfound).save(imageFN+'.bin.notFoundPixels.tif')
	# Image.fromarray(ndimg.gaussian_filter(im_notfound,gaussWidth)).save(imageFN+'.bin.notFoundPixelsBlurred.tif')

for iF in imageFiles:
	fparams = open(configFile,'w')
	for line in lines:
		if line.startswith('DoFwd'): fparams.write(doFwd)
		else: fparams.write(line)
	fparams.close()
	runFile(iF)
	doFwd = 'DoFwd 0\n'
