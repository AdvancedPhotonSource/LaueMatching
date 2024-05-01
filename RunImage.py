
# imageFile = ['test_3.h5','test_4.h5']
# configFile = 'params_ni.txt'
# nMax = 5 #Max # Grains expected. This is used for coloring

# imageFile = ['test_5_cleaned.h5']
# configFile = 'params_al_cleaned.txt'
# nMax = 50 #Max # Grains expected. This is used for coloring

# imageFile = ['test_6_cleaned.h5']
# configFile = 'params_EuAl2O4.txt'
# nMax = 5 #Max # Grains expected. This is used for coloring

imageFile = ['simulated_data.h5']
configFile = 'params_sim.txt'
nMax = 50

import h5py
import numpy as np
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
import math
import os
import subprocess
import shutil
from PIL import Image
import pickle as pl
import skimage
plt.rcParams['font.size'] = 3

colors = plt.get_cmap('nipy_spectral',nMax)

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
	h_image = h5py.File(imageFN,'r')
	imageFN = resultdir + '/'+ imageFN
	hf_out = h5py.File(imageFN+'.bin.output.h5','w')
	print(f'Processing file {imageFN}.bin')
	h_im = np.array(h_image['entry1']['data']['data'][()])
	hf_out.create_dataset('/entry/data/cleaned_data',data=h_im)
	h_im[h_im<thresh] = 0
	hf_out.create_dataset('/entry/data/cleaned_data_threshold',data=h_im)

	# Do connected components and check for size of spots
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
	Image.fromarray(h_im).save(imageFN+'.bin.input.tif')
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

	# Alternatively, do watershed instead of ndimg.label
	labels = skimage.segmentation.watershed(-h_im,mask=h_im,connectivity=2)
	nlabels = np.max(labels)

	labels2 = skimage.segmentation.watershed(-h_im2,mask=h_im2,connectivity=2)
	nlabels2 = np.max(labels2)

	h_im2.astype(np.double).tofile(imageFN+'.bin')
	hf_out.create_dataset('/entry/data/input_blurred',data=h_im2)
	Image.fromarray(h_im2).save(imageFN+'.bin.inputBlurred.tif')
	fout = open(imageFN+'.laue_dict_output.txt','w')
	subprocess.call('./laue_dict_faster '+configFile+' '+orientf+' '+hklf+' '+imageFN+'.bin 200',shell=True,
				stdout=fout)
	fout.close()
	grainInfo = np.genfromtxt(imageFN+'.bin.solutions.txt',skip_header=1)
	headgr = open(imageFN+'.bin.solutions.txt').readline()
	spotInfo = np.genfromtxt(imageFN+'.bin.spots.txt',skip_header=1)
	headsp = open(imageFN+'.bin.spots.txt').readline()
	or_gr = hf_out.create_dataset('/entry/results/original_grains',data=grainInfo)
	or_gr.attrs['head'] = np.string_(headgr)
	or_sp = hf_out.create_dataset('/entry/results/original_spots',data=spotInfo)
	or_sp.attrs['head'] = np.string_(headsp)
	outfgr = open(imageFN+'.bin.good_solutions.txt','a')
	outfsp = open(imageFN+'.bin.good_spots.txt','a')
	outfgr.write(headgr)
	outfsp.write(headsp)
	grains = []
	spots = []
	legends=[]
	nsols = 0
	label_found = []
	fmtout = '%4d\t%4d\t%11.2f\t%11.2f\t%12.3f\t%4d\t%4d\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t'
	fmtout += '%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f'
	fmtout += '\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%12.6f\t%9.6f\t%10d'
	if (len(grainInfo.shape)==2): grainInfo = grainInfo[np.argsort(-grainInfo[:,4])]
	else: grainInfo = np.expand_dims(grainInfo,axis=0)
	# Make a figure with the image on the background
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig,[0.,0.,1.,1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	# fig,ax = plt.subplots()
	h_im_plt = h_im
	h_im_plt[h_im_plt==0] = 1
	ax.imshow(np.log(h_im_plt),cmap='Greens')
	grainNr = 0
	for grain in grainInfo:
		spotThis = spotInfo[spotInfo[:,0]==grain[0],:]
		goodSpots = spotThis[spotThis[:,-1]>=thresh,:]
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
				ax.text(spot[5]-np.random.randint(0,20),spot[6]-20,indices,c=colors(grainNr))
				if labels2[int(spot[6])][int(spot[5])]:
					label_found.append(labels2[int(spot[6])][int(spot[5])])
			# We need to check for the spots that were overlapping, but in the gaussian blurred image
			grain[5] = goodSpots.shape[0]
			thisID = grain[0]
			nsols+=1
			grain = grain.reshape(grain.shape[0],-1).transpose()
			np.savetxt(outfgr,grain,fmt=fmtout)
			np.savetxt(outfsp,goodSpots,fmt='%4d\t%3d\t%3d\t%3d\t%3d\t%5d\t%5d\t%9.6f\t%9.6f\t%9.6f\t%7d')
			# Save an image with the blobs from found spots as open squares and grain id
			lbl = 'GrainID '+str(int(thisID))
			ax.plot(goodSpots[:,5],goodSpots[:,6],'ks', markerfacecolor='none', ms=3, markeredgecolor=colors(grainNr),markeredgewidth=0.1,label=lbl)
			grainNr+=1
	plt.legend()
	# with open(imageFN+'.bin.LabeledImage.pkl','wb') as pf:
	# 	pl.dump(fig,pf)
	plt.savefig(imageFN+'.bin.LabeledImage.tif',dpi=outdpi)
	l_im = np.array(Image.open(imageFN+'.bin.LabeledImage.tif').convert('RGB'))
	hf_out.create_dataset('/entry/results/LabeledImage',data=l_im)
	outfgr.close()
	outfsp.close()
	goodGrs = np.genfromtxt(imageFN+'.bin.good_solutions.txt',skip_header=1)
	goodSps = np.genfromtxt(imageFN+'.bin.good_spots.txt',skip_header=1)
	go_gr = hf_out.create_dataset('/entry/results/filtered_grains',data=goodGrs)
	go_gr.attrs['head'] = np.string_(headgr)
	go_sp = hf_out.create_dataset('/entry/results/filtered_spots',data=goodSps)
	go_sp.attrs['head'] = np.string_(headsp)
	# Find the labels found, remove them and save a figure with those spots only.
	label_found = set(label_found)
	print(f'Blurring width of {gaussWidth} pixels was used.')
	print(f'{len(centers)} peaks in original image, {len(label_found)} successfully identified.')
	print(f'{nsols} grains were found after filtering out intensity and duplicate matches.')
	im_notfound = np.copy(h_im)
	im_found = np.zeros((h_im.shape))
	for label in label_found:
		im_found[labels2==label] = 255
		im_notfound[labels2==label] = 0
	hf_out.create_dataset('/entry/results/found_pixels',data=im_found)
	hf_out.create_dataset('/entry/results/not_found_pixels',data=im_notfound)
	hf_out.create_dataset('/entry/results/not_found_pixels_blurred',data=ndimg.gaussian_filter(im_notfound,gaussWidth))
	hf_out.close()
	Image.fromarray(im_found).save(imageFN+'.bin.FoundPixels.tif')
	Image.fromarray(im_notfound).save(imageFN+'.bin.notFoundPixels.tif')
	Image.fromarray(ndimg.gaussian_filter(im_notfound,gaussWidth)).save(imageFN+'.bin.notFoundPixelsBlurred.tif')

lines = open(configFile,'r').readlines()
numRuns = 1
for line in lines:
	if line.startswith('SpaceGroup'):
		sgNum = int(line.split()[1])
	if line.startswith('nRuns'):
		numRuns = int(line.split()[1])
	elif line.startswith('P_Array'):
		distt = float(line.split()[3])
	elif line.startswith('Threshold'):
		thresh = float(line.split()[1])
	elif line.startswith('MinIntensity'):
		minIntensity = float(line.split()[1])
	elif line.startswith('PxX'):
		px = float(line.split()[1])
	elif line.startswith('OrientationSpacing'):
		orientSpacing = float(line.split()[1])
	elif line.startswith('NrPxX'):
		nrPixels = int(line.split()[1])
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
	elif line.startswith('ForwardFile'):
		fwdf = line.split()[1]

outdpi = 600
scalarX = nrPixels/outdpi
scalarY = nrPixels/outdpi
plt.rcParams['figure.figsize'] = [scalarX, scalarY]

if os.path.exists(fwdf):
	size_orient = os.path.getsize(orientf)
	nrOrient = int(size_orient/(8*9))
	size_fwd = os.path.getsize(fwdf)
	nrOrientFwd = int(size_fwd/(2*(1+2*maxLaueSpots)))
	if nrOrient != nrOrientFwd: doFwd = 'DoFwd 1\n'
	else: doFwd = 'DoFwd 0\n'
else:
	doFwd = 'DoFwd 1\n'

if os.path.exists(resultdir):
	shutil.rmtree(resultdir,ignore_errors=True)
os.makedirs(resultdir,exist_ok=True)

print("Laue image processing using dictionary")

for iF in imageFile:
	fparams = open(configFile,'w')
	for line in lines:
		if line.startswith('DoFwd'): fparams.write(doFwd)
		else: fparams.write(line)
	fparams.close()
	runFile(iF)
	doFwd = 'DoFwd 0\n'
