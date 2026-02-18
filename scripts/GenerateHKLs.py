import sys
import math
import numpy as np
from math import cos, sin, sqrt
import argparse
import warnings

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

class LaueMatching:
    """Class for computing valid reflections in Laue diffraction."""

    def __init__(self, args):
        """Initialize with command line arguments."""
        self.resultFileName = args.resultFileName
        self.sym = args.sym
        self.latticeParameter = args.latticeParameter
        self.sgNum = args.sgnum
        self.R = np.asarray(args.RArray)
        self.P = np.asarray(args.PArray)
        self.Nx = args.NumPxX
        self.Ny = args.NumPxY
        self.dx = args.dx
        self.dy = args.dy
        
        # Constants
        self.hc_keVnm = 1.2398419739
        self.rad2deg = 180/np.pi
        self.deg2rad = np.pi/180
        
        # Derived properties
        self.usingHexAxes = (abs(90-self.latticeParameter[3]) + 
                            abs(90-self.latticeParameter[4]) + 
                            abs(120-self.latticeParameter[5])) < 1e-6
        
        self.Hexagonal = 1 if (self.sgNum >= 168 and self.sgNum < 195) else 0
        
        # Validation
        if self.sym not in 'FICARPB' or len(self.sym) != 1:
            print('Invalid value for sym, must be one character from F,I,C,A,R,P,B')
            sys.exit(1)

    def zeroOut(self, val):
        """Set very small values to zero to avoid floating point errors."""
        if np.abs(val) < 1e-11:
            return 0
        else:
            return val

    def calcRecipArray(self):
        """Calculate the reciprocal lattice array from lattice parameters."""
        recip = np.zeros((3, 3))
        Lat = self.latticeParameter
        a = Lat[0]; b = Lat[1]; c = Lat[2]
        alpha = Lat[3]; beta = Lat[4]; gamma = Lat[5]
        
        rhomb = 0
        if (self.sgNum in [146, 148, 155, 160, 161, 166, 167]):
            rhomb = 1
            
        ca = cos(alpha * self.deg2rad)
        cb = cos(beta * self.deg2rad)
        cg = cos(gamma * self.deg2rad)
        sg = sin(gamma * self.deg2rad)
        phi = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
        Vc = a*b*c * phi
        pv = (2*np.pi) / (Vc)
        
        if rhomb == 0:
            a0 = a             ; a1 = 0.0               ; a2 = 0.0
            b0 = b*cg          ; b1 = b*sg              ; b2 = 0
            c0 = c*cb          ; c1 = c*(ca-cb*cg)/sg   ; c2 = c*phi/sg
            a0 = self.zeroOut(a0); a1 = self.zeroOut(a1); a2 = self.zeroOut(a2)
            b1 = self.zeroOut(b1); b2 = self.zeroOut(b2)
            c2 = self.zeroOut(c2)
        else:
            p = sqrt(1.0 + 2*ca)
            q = sqrt(1.0 - ca)
            pmq = (a/3.0)*(p-q)
            p2q = (a/3.0)*(p+2*q)
            a0 = p2q; a1 = pmq; a2 = pmq
            b0 = pmq; b1 = p2q; b2 = pmq
            c0 = pmq; c1 = pmq; c2 = p2q
            
        recip[0][0] = self.zeroOut((b1*c2-b2*c1)*pv)
        recip[1][0] = self.zeroOut((b2*c0-b0*c2)*pv)
        recip[2][0] = self.zeroOut((b0*c1-b1*c0)*pv)
        recip[0][1] = self.zeroOut((c1*a2-c2*a1)*pv)
        recip[1][1] = self.zeroOut((c2*a0-c0*a2)*pv)
        recip[2][1] = self.zeroOut((c0*a1-c1*a0)*pv)
        recip[0][2] = self.zeroOut((a1*b2-a2*b1)*pv)
        recip[1][2] = self.zeroOut((a2*b0-a0*b2)*pv)
        recip[2][2] = self.zeroOut((a0*b1-a1*b0)*pv)
        
        return recip

    def _ALLOW_FC(self, h, k, l):
        """Face-centered, hkl must be all even or all odd."""
        return (h+k) % 2 == 0 and (k+l) % 2 == 0

    def _ALLOW_BC(self, h, k, l):
        """Body-centered, sum must be even."""
        return (h+k+l) % 2 == 0

    def _ALLOW_CC(self, h, k, l):
        """C-centered."""
        return (h+k) % 2 == 0

    def _ALLOW_AC(self, h, k, l):
        """A-centered."""
        return (k+l) % 2 == 0

    def _ALLOW_RHOM_HEX(self, h, k, l):
        """Rhombohedral hexagonal, allowed are -H+K+L=3n or H-K+L=3n."""
        return (-h+k+l) % 3 == 0 or (h-k+l) % 3 == 0

    def _ALLOW_HEXAGONAL(self, h, k, l):
        """Hexagonal, forbidden are: H+2K=3N with L odd."""
        return bool((h+2*k) % 3) or not bool(l % 2)

    def createDetector(self):
        """Create a detector object with the specified parameters."""
        return DetectorType(Nx=self.Nx, Ny=self.Ny, dx=self.dx, dy=self.dy, 
                           R=self.R, P=self.P, name='Perkn-Elmer')
    def preCalcHKLs(self, detector, recip, Ehi=30):
        """
        Pre-calculate valid HKLs based on detector and lattice parameters.
        
        Parameters:
            detector: DetectorType object containing detector geometry
            recip: Reciprocal lattice matrix
            Ehi: Maximum energy in keV (default: 30)
            
        Returns:
            numpy array of valid HKLs with their squared magnitudes
        """
        LatC = self.latticeParameter
        
        # Calculate the maximum scattering vector magnitude
        XYZ = detector.pixel2XYZ((detector.Nx-1)/2, detector.Ny-1)
        if XYZ is None:
            raise ValueError("Could not compute detector coordinates")
            
        thMax = math.atan2(XYZ.item((0,1)), XYZ.item((0,2))) / 2
        Qmax = 4*math.pi * math.sin(thMax) * Ehi/self.hc_keVnm
        
        # Calculate maximum indices based on lattice parameters
        hmax = int(math.floor(Qmax / (2*math.pi / LatC[0])))
        kmax = int(math.floor(Qmax / (2*math.pi / LatC[1])))
        lmax = int(math.floor(Qmax / (2*math.pi / LatC[2])))
        
        # Create arrays of all possible h, k, l values
        h_range = np.arange(-hmax, hmax+1)
        k_range = np.arange(-kmax, kmax+1)
        l_range = np.arange(-lmax, lmax+1)
        
        # Generate all possible combinations (this is more memory efficient than triple nested loops)
        h_vals, k_vals, l_vals = np.meshgrid(h_range, k_range, l_range, indexing='ij')
        
        # Flatten the arrays
        h_flat = h_vals.flatten()
        k_flat = k_vals.flatten()
        l_flat = l_vals.flatten()
        
        # Create a mask to filter valid reflections
        # First, remove the origin (0,0,0)
        valid_mask = ~((h_flat == 0) & (k_flat == 0) & (l_flat == 0))
        
        # Apply symmetry constraints
        if self.sym == 'F':
            # Face-centered: h,k,l must be all even or all odd
            valid_mask &= ((h_flat + k_flat) % 2 == 0) & ((k_flat + l_flat) % 2 == 0)
        elif self.sym == 'I':
            # Body-centered: h+k+l must be even
            valid_mask &= ((h_flat + k_flat + l_flat) % 2 == 0)
        elif self.sym == 'C':
            # C-centered: h+k must be even
            valid_mask &= ((h_flat + k_flat) % 2 == 0)
        elif self.sym == 'A':
            # A-centered: k+l must be even
            valid_mask &= ((k_flat + l_flat) % 2 == 0)
        elif self.sym == 'R' and self.usingHexAxes:
            # Rhombohedral hexagonal: -h+k+l=3n or h-k+l=3n
            valid_mask &= ((-h_flat + k_flat + l_flat) % 3 == 0) | ((h_flat - k_flat + l_flat) % 3 == 0)
        
        # Additional hexagonal constraints if needed
        if self.Hexagonal:
            valid_mask &= ((h_flat + 2*k_flat) % 3 != 0) | (l_flat % 2 == 0)
        
        # Apply the mask to get valid indices
        h_valid = h_flat[valid_mask]
        k_valid = k_flat[valid_mask]
        l_valid = l_flat[valid_mask]
        
        # Calculate the squared magnitudes
        magnitude_squared = h_valid**2 + k_valid**2 + l_valid**2
        
        # Create the output array
        num_valid = len(h_valid)
        hklarr = np.zeros((num_valid, 4))
        hklarr[:, 0] = h_valid
        hklarr[:, 1] = k_valid
        hklarr[:, 2] = l_valid
        hklarr[:, 3] = magnitude_squared
        
        return hklarr

    def run(self):
        """Execute the main workflow to generate and save valid HKLs."""
        try:
            # Create detector
            detector = self.createDetector()
            
            # Calculate reciprocal lattice
            recip = self.calcRecipArray()
            
            # Apply random orientation
            om = np.array([
                [0.8340462, -0.54723977, 0.06996829],
                [0.51044064, 0.71732763, -0.47422719],
                [0.20932579, 0.43124205, 0.87761781]
            ])
            recip = recip.dot(om)
            
            # Calculate valid HKLs
            hklarr = self.preCalcHKLs(detector, recip)
            print(f'Number of valid HKLs: {hklarr.shape[0]}')
            
            # Sort by length and save to file
            hklarr2 = hklarr[hklarr[:, 3].argsort()]
            np.savetxt(self.resultFileName, hklarr2, fmt="%d %d %d %d")
            print(f'Results saved to: {self.resultFileName}')
            
        except Exception as e:
            print(f"Error during execution: {e}")
            sys.exit(1)


class DetectorType:
    """Class representing a detector configuration."""
    
    def __init__(self, Nx, Ny, dx, dy, R, P, name=''):
        """
        Initialize detector with given parameters.
        
        Parameters:
        Nx, Ny: Number of pixels in X and Y directions
        dx, dy: Pixel size in X and Y directions [m]
        R: Rotation array describing detector orientation [degrees]
        P: Translation array describing detector position [m]
        name: Detector name (optional)
        """
        try:
            if Nx <= 2: 
                raise ValueError('Nx = ' + repr(Nx))
            if Ny <= 2: 
                raise ValueError('Ny = ' + repr(Ny))
                
            Nx = round(Nx)
            Ny = round(Ny)

            if dx <= 0: 
                raise ValueError('dx = ' + repr(dx))
            if dy <= 0: 
                raise ValueError('dy = ' + repr(dy))

        except ValueError as e:
            print(f"Invalid detector parameters: {e}")
            sys.exit(1)

        self.name = name
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.P = np.asarray([P[0], P[1], P[2]])
        
        # Calculate rotation matrix
        rotang = np.linalg.norm(R)
        if rotang > 0:  # Avoid division by zero
            rotvect = R/np.linalg.norm(R)
            
            # Rodriguez formula for rotation matrix
            self.rot = np.matrix([
                [math.cos(rotang)+(1-math.cos(rotang))*(rotvect[0]**2), 
                 (1-math.cos(rotang))*rotvect[0]*rotvect[1]-math.sin(rotang)*rotvect[2], 
                 (1-math.cos(rotang))*rotvect[0]*rotvect[2]+math.sin(rotang)*rotvect[1]],
                [(1-math.cos(rotang))*rotvect[1]*rotvect[0]+math.sin(rotang)*rotvect[2], 
                 math.cos(rotang)+(1-math.cos(rotang))*(rotvect[1]**2),  
                 (1-math.cos(rotang))*rotvect[1]*rotvect[2]-math.sin(rotang)*rotvect[0]],
                [(1-math.cos(rotang))*rotvect[2]*rotvect[0]-math.sin(rotang)*rotvect[1], 
                 (1-math.cos(rotang))*rotvect[2]*rotvect[1]+math.sin(rotang)*rotvect[0], 
                 math.cos(rotang)+(1-math.cos(rotang))*(rotvect[2]**2)]
            ])
        else:
            # Identity matrix for zero rotation
            self.rot = np.matrix(np.identity(3))
            
        self.pCenter = ((self.Nx-1)/2, (self.Ny-1)/2)  # center of detector (pixels)
        self.XYZcenter = self.pixel2XYZ(self.pCenter[0], self.pCenter[1])  # center in beam-line coords

    def pixel2XYZ(self, px, py):
        """Convert pixel coordinates to beam-line coordinates XYZ."""
        try:
            px = np.double(px)
            py = np.double(py)
        except:
            raise ValueError('Cannot interpret pixel: ' + repr(px) + '  ' + repr(py))
            
        if px < 0 or px >= self.Nx: 
            return None  # must lie on detector
        if py < 0 or py >= self.Ny: 
            return None

        # x' and y' (requiring z'=0), detector starts centered on origin and perpendicular to z-axis
        xp = (px - 0.5*(self.Nx-1)) * self.dx  # (x' y' z'), position on detector
        yp = (py - 0.5*(self.Ny-1)) * self.dy
        xp += self.P[0]  # translate by P
        yp += self.P[1]
        zp = self.P[2]
        xyz = np.matrix([xp, yp, zp])  # position in detector frame
        XYZ = self.rot.dot(xyz.T).flatten()
        return XYZ

    def XYZ2pixel(self, XYZ):
        """Convert beam-line coordinates XYZ to pixel coordinates."""
        try:
            if XYZ.shape != (1, 3): 
                return None
        except: 
            return None
            
        xyz = np.linalg.inv(self.rot).dot(XYZ.T).T  # un-rotate

        z = xyz.item(0, 2)
        if z <= 0: 
            return None  # does not point to detector
            
        scale = self.P[2] / z  # scale xyz so that z = dist
        xyz = xyz * scale

        xp = xyz.item(0, 0) - self.P[0]  # un-translate by P[]
        yp = xyz.item(0, 1) - self.P[1]
        px = xp / self.dx + 0.5*(self.Nx-1)  # (x' y' z') position on detector --> pixel
        py = yp / self.dy + 0.5*(self.Ny-1)

        if px < 0 or px >= (self.Nx-1): 
            return None  # must land on detector
        if py < 0 or py >= (self.Ny-1): 
            return None

        return (px, py)


def main():
    """Parse arguments and run the LaueMatching workflow."""
    warnings.filterwarnings('ignore')
    
    parser = MyParser(description='''
    LaueMatching compute valid reflections.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-resultFileName', type=str, required=False, 
                      default='valid_reflections.csv', help='Name of output file')
    parser.add_argument('-sym', type=str, required=False, default='F', 
                      help='First character of the Hermann-Maguin description. Accepted values: F,I,C,A,R,P,B')
    parser.add_argument('-sgnum', type=int, required=False, default=225, 
                      help='SpaceGroup number. Between 1-230.')
    parser.add_argument('-latticeParameter', type=float, nargs=6, required=True, 
                      help='Lattice Parameter a,b,c,alpha,beta,gamma [nm, nm, nm, degrees, degrees, degrees].')
    parser.add_argument('-RArray', type=float, nargs=3, required=True, 
                      help='Rotation array describing detector orientation. 3 values.[degrees]')
    parser.add_argument('-PArray', type=float, nargs=3, required=True, 
                      help='Translation array describing detector position. 3 values.[m]')
    parser.add_argument('-NumPxX', type=float, required=False, default=2048, 
                      help='Number of pixels in X direction.')
    parser.add_argument('-NumPxY', type=float, required=False, default=2048, 
                      help='Number of pixels in Y direction.')
    parser.add_argument('-dx', type=float, required=False, default=200e-6, 
                      help='Pixel size in X direction.[m]')
    parser.add_argument('-dy', type=float, required=False, default=200e-6, 
                      help='Pixel size in Y direction.[m]')
    
    args, unparsed = parser.parse_known_args()
    
    # Create and run the LaueMatching instance
    laue = LaueMatching(args)
    laue.run()


if __name__ == "__main__":
    main()