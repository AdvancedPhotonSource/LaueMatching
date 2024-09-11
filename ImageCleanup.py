import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import sys
import scipy.ndimage as ndimg
from skimage.measure import label, regionprops
import argparse
import diplib as dip
import matplotlib.pyplot as plt
from PIL import Image

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def runMedian(img,rad):
      return dip.MedianFilter(img,rad)
parser = MyParser(description='''
LaueMatching Image Cleanup.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-inputFile', type=str, required=True, help='Name of input h5 file in the dataExchange format.')
parser.add_argument('-minArea', type=int, required=False, default=10, help='Minimum area of connected pixels qualifying signal.')
parser.add_argument('-maxArea', type=int, required=False, default=1000, help='Maximum area of connected pixels qualifying signal.')
parser.add_argument('-radius', type=int, required=False, default=101, help='Radius of median filter to apply to compute background.')
parser.add_argument('-nPasses', type=int, required=False, default=5, help='Number of median filter passes to compute background.')
parser.add_argument('-imageType', type=str, required=False, default='hdf', help='Image type option: hdf, tiff or jpg.')
args, unparsed = parser.parse_known_args()
fn = args.inputFile
minArea = args.minArea
maxArea = args.maxArea
rad = args.radius
nPasses = args.nPasses
im_type = args.imageType

if im_type == 'hdf':
    fnout = fn.split('.')[0]+'_cleaned.'+fn.split('.')[1]
    hf = h5py.File(fn,'r')
    img = np.array(hf['entry1/data/data'][()]).astype(np.uint16)
    hf.close()
else:
    fnout = fn.split('.')[0]+'_cleaned.'+fn.split('.')[1]+'.h5'
    img = (np.array(Image.open(fn)).astype(np.uint16)[:,:,0]*(-1) + 255)
    img *= int(16000/255)

print(fnout)
data2 = dip.Image(img)
for i in range(nPasses):
	data2 = runMedian(data2,rad)

im_corr = img.astype(np.double)-data2
thresh = 60*(1+np.std(im_corr)//60)
print(f'Computed threshold: {thresh}')
im_corr[im_corr<thresh] = 0
im_corr = im_corr.astype(np.uint16)
labels,nlabels = ndimg.label(im_corr)
for lNr in range(1,nlabels):
    if (np.sum(labels==lNr) < minArea):
        im_corr[labels==lNr] = 0
        labels[labels==lNr] = 0
    if (np.sum(labels==lNr) > maxArea):
        im_corr[labels==lNr] = 0
        labels[labels==lNr] = 0

fig = plt.figure(frameon=False)
ax = plt.Axes(fig,[0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.log(im_corr+1),cmap='gray_r')
props = regionprops(labels)
for prop in props:
    centroid = prop.centroid
    ax.plot(centroid[1],centroid[0],'ks', markerfacecolor='none', ms=10, markeredgecolor='black',markeredgewidth=1)
ax.axis('equal')
plt.show()


hf = h5py.File(fnout,'w')
hf.create_dataset('/entry1/data/data',data=im_corr)
hf.create_dataset('/entry1/data/rawdata',data=img)
hf.create_dataset('/entry1/data/background',data=data2)
hf.create_dataset('/entry1/data/threshold',data=thresh)
hf.close()
