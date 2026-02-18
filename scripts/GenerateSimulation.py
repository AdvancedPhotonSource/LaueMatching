#!/usr/bin/env python
"""
LaueMatching: Generate Simulation using an experiment configuration and a list of orientations.
This script generates simulated diffraction patterns based on input configurations.

Contact: hsharma@anl.gov
"""

import numpy as np
import h5py
from math import pi, sin, cos
import scipy.ndimage as ndimg
from PIL import Image
import argparse
import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
HC_KEV_NM = 1.2398419739  # Planck's constant * speed of light in keV*nm

class ConfigParser:
    """Parser for the configuration file with all simulation parameters."""
    
    def __init__(self, config_file):
        """
        Initialize the parser with a configuration file path.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.params = {
            'sgNum': None,
            'sym': 'F',
            'latC': None,
            'r_arr': None,
            'p_arr': None,
            'dx': None,
            'dy': None,
            'astar': -1,
            'Elo': None,
            'Ehi': None,
            'gaussWidth': None,
            'nPxX': None,
            'nPxY': None,
            'hklf': 'valid_reflections.csv'
        }
        self._parse_config()
        
    def _parse_config(self):
        """Parse the configuration file and extract parameters."""
        try:
            with open(self.config_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.startswith('SpaceGroup'):
                    self.params['sgNum'] = int(line.split()[1])
                elif line.startswith('Symmetry'):
                    sym = line.split()[1]
                    if sym not in 'FICAR' and len(sym) != 1:
                        raise ValueError('Invalid value for sym, must be one character from F,I,C,A,R')
                    self.params['sym'] = sym
                elif line.startswith('LatticeParameter'):
                    self.params['latC'] = ' '.join(line.split()[1:7])
                elif line.startswith('R_Array'):
                    self.params['r_arr'] = ' '.join(line.split()[1:4])
                elif line.startswith('P_Array'):
                    self.params['p_arr'] = ' '.join(line.split()[1:4])
                elif line.startswith('PxX'):
                    self.params['dx'] = float(line.split()[1])
                elif line.startswith('PxY'):
                    self.params['dy'] = float(line.split()[1])
                elif line.startswith('AStar'):
                    self.params['astar'] = float(line.split()[1])
                elif line.startswith('Elo'):
                    self.params['Elo'] = float(line.split()[1])
                elif line.startswith('Ehi'):
                    self.params['Ehi'] = float(line.split()[1])
                elif line.startswith('SimulationSmoothingWidth'):
                    self.params['gaussWidth'] = int(line.split()[1])
                elif line.startswith('NrPxX'):
                    self.params['nPxX'] = int(line.split()[1])
                elif line.startswith('NrPxY'):
                    self.params['nPxY'] = int(line.split()[1])
                elif line.startswith('HKLFile'):
                    self.params['hklf'] = line.split()[1]
                    
            # Validate required parameters
            required = ['sgNum', 'latC', 'r_arr', 'p_arr', 'dx', 'dy', 'Elo', 'Ehi', 
                        'gaussWidth', 'nPxX', 'nPxY']
            
            for param in required:
                if self.params[param] is None:
                    raise ValueError(f"Missing required parameter: {param}")
                    
            # Set derived parameters
            self.params['P'] = np.array([float(p) for p in self.params['p_arr'].split()])
            self.params['R'] = np.array([float(r) for r in self.params['r_arr'].split()])
            
            # If astar is not provided, calculate from the first lattice parameter
            if self.params['astar'] == -1:
                self.params['astar'] = 2*pi/float(self.params['latC'].split()[0])
                
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_file} not found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            sys.exit(1)
            
    def get_params(self):
        """Return the parsed parameters."""
        return self.params


class OrientationLoader:
    """Load orientation matrices from a file."""
    
    def __init__(self, orientation_file):
        """
        Initialize the loader with an orientation file path.
        
        Args:
            orientation_file (str): Path to the file containing orientations
        """
        self.orientation_file = orientation_file
        
    def load_orientations(self):
        """
        Load orientation matrices from the file.
        
        Returns:
            numpy.ndarray: Array of orientation matrices
        """
        try:
            with open(self.orientation_file) as f:
                lines = f.readlines()
                
            if lines[0].startswith('%GrainNr'):
                orientations = np.genfromtxt(self.orientation_file, skip_header=1)[:, 22:31]
            else:
                orientations = np.genfromtxt(self.orientation_file)
                
            return orientations
        except FileNotFoundError:
            logger.error(f"Orientation file {self.orientation_file} not found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading orientations: {str(e)}")
            sys.exit(1)


class DiffractionSimulator:
    """Simulate diffraction patterns based on orientations and configuration."""
    
    def __init__(self, params, hkl_data, skip_percentage=30.0):
        """
        Initialize the simulator with parameters and HKL data.
        
        Args:
            params (dict): Configuration parameters
            hkl_data (numpy.ndarray): HKL reflection data
            skip_percentage (float): Percentage of spots to randomly skip (0-100)
        """
        self.params = params
        self.hkl_data = hkl_data
        self.skip_percentage = skip_percentage
        
        # Validate skip_percentage
        if not 0 <= self.skip_percentage <= 100:
            logger.warning(f"Invalid skip_percentage: {self.skip_percentage}. Using default 30.0%.")
            self.skip_percentage = 30.0
            
        logger.info(f"Spot skipping percentage set to: {self.skip_percentage}%")
        
        # Calculate skip threshold (0-10 scale for random number generation)
        self.skip_threshold = int(self.skip_percentage / 10.0)
        
        # Initialize rotation matrix
        self._init_rotation_matrix()
        
        # Initialize arrays for simulation
        self.img = np.zeros((self.params['nPxX'], self.params['nPxY']))
        self.pos_arr = []
        self.ki = np.array([0, 0, 1.0])
        
    def _init_rotation_matrix(self):
        """Initialize the rotation matrix based on R parameters."""
        R = self.params['R']
        rotang = np.linalg.norm(R)
        rotvect = R / np.linalg.norm(R)
        
        # Rotation matrix using Rodrigues' rotation formula
        self.rot = np.array([
            [cos(rotang) + (1 - cos(rotang)) * (rotvect[0] ** 2),
             (1 - cos(rotang)) * rotvect[0] * rotvect[1] - sin(rotang) * rotvect[2],
             (1 - cos(rotang)) * rotvect[0] * rotvect[2] + sin(rotang) * rotvect[1]],
            [(1 - cos(rotang)) * rotvect[1] * rotvect[0] + sin(rotang) * rotvect[2],
             cos(rotang) + (1 - cos(rotang)) * (rotvect[1] ** 2),
             (1 - cos(rotang)) * rotvect[1] * rotvect[2] - sin(rotang) * rotvect[0]],
            [(1 - cos(rotang)) * rotvect[2] * rotvect[0] - sin(rotang) * rotvect[1],
             (1 - cos(rotang)) * rotvect[2] * rotvect[1] + sin(rotang) * rotvect[0],
             cos(rotang) + (1 - cos(rotang)) * (rotvect[2] ** 2)]
        ])
    
    def xyz_to_pixel(self, xyz):
        """
        Convert XYZ coordinates to detector pixel coordinates.
        
        Args:
            xyz (numpy.ndarray): 3D coordinates
            
        Returns:
            tuple or None: (px, py) pixel coordinates or None if outside detector
        """
        xyz_transformed = np.linalg.inv(self.rot).dot(xyz.T).T
        z = xyz_transformed[0, 2]
        
        if z <= 0:
            return None
            
        scale = self.params['P'][2] / z
        xyz_scaled = xyz_transformed * scale
        
        xp = xyz_scaled[0, 0] - self.params['P'][0]
        yp = xyz_scaled[0, 1] - self.params['P'][1]
        
        px = xp / self.params['dx'] + 0.5 * (self.params['nPxX'] - 1)
        py = yp / self.params['dy'] + 0.5 * (self.params['nPxY'] - 1)
        
        if px < 0 or px >= (self.params['nPxX'] - 1):
            return None
        if py < 0 or py >= (self.params['nPxY'] - 1):
            return None
            
        return (px, py)
    
    def get_spots(self, recip, grain_nr):
        """
        Calculate diffraction spots for a given reciprocal space orientation.
        
        Args:
            recip (numpy.ndarray): Reciprocal space orientation matrix
            grain_nr (int): Grain number for tracking
            
        Returns:
            int: Number of spots generated
        """
        # Extract HKL indices
        hkls = np.copy(self.hkl_data[:, :3])
        
        # Calculate q-vectors
        qvecs = recip.dot(hkls.T).T
        qlens = np.linalg.norm(qvecs, axis=1)
        
        # Filter valid q-vectors
        good = qlens > 0
        qvecs = qvecs[good, :]
        qlens = qlens[good]
        
        # Calculate normalized q-vectors
        qhats = np.divide(qvecs, np.column_stack([qlens, qlens, qlens]))
        
        # Calculate dot products
        dots = np.squeeze(np.copy(qhats[:, 2]))
        mdots = np.empty(qhats.shape)
        mdots[:, 0] = np.copy(dots)
        mdots[:, 1] = np.copy(dots)
        mdots[:, 2] = np.copy(dots)
        
        # Calculate k-vectors
        kfs = self.ki - 2 * np.multiply(mdots, qhats)
        
        # Transform to detector space
        xyz = np.linalg.inv(self.rot).dot(kfs.T).T
        z = np.squeeze(np.copy(xyz[:, 2]))
        
        # Filter points with positive z
        goodZ = z > 0
        qlens = qlens[goodZ]
        qhats = qhats[goodZ, :]
        xyz = xyz[goodZ, :]
        
        # Scale by detector distance
        xyz = np.divide(xyz * self.params['P'][2], xyz[:, 2, np.newaxis])
        
        # Calculate detector coordinates
        xp = xyz[:, 0] - self.params['P'][0]
        yp = xyz[:, 1] - self.params['P'][1]
        
        px = xp / self.params['dx'] + 0.5 * (self.params['nPxX'] - 1)
        py = yp / self.params['dy'] + 0.5 * (self.params['nPxY'] - 1)
        
        # Filter points within detector bounds
        good = (px >= 0) & (px < (self.params['nPxX'] - 1)) & (py >= 0) & (py < (self.params['nPxY'] - 1))
        px = px[good]
        py = py[good]
        
        # Compile pixel coordinates
        pixels = np.vstack((px, py)).T
        
        # Filter by q-length
        qlens = qlens[good]
        qhats = qhats[good, :]
        
        # Calculate energies
        sin_thetas = -qhats[:, 2]
        energies = HC_KEV_NM * np.divide(qlens, sin_thetas) / (4 * pi)
        
        # Filter by energy range
        good_e = (energies > self.params['Elo']) & (energies < self.params['Ehi'])
        pixels = pixels[good_e, :]
        
        # Add spots to image and position array
        nr_pixels = 0
        for pixel in pixels:
            self.pos_arr.append([pixel[1], pixel[0], grain_nr])
            
            # Skip spots based on skip_percentage
            # If skip_percentage is 0, don't skip any spots
            if self.skip_percentage == 0 or np.random.randint(0, 10) >= self.skip_threshold:
                nr_pixels += 1
                self.img[int(pixel[1]), int(pixel[0])] = np.random.randint(500, 16000)
                self.pos_arr[-1].append(1)
            else:
                self.pos_arr[-1].append(0)
                
        logger.info(f'Grain {grain_nr}: Generated {nr_pixels} spots')
        return nr_pixels
    
    def simulate(self, orientations):
        """
        Run the simulation for all orientations.
        
        Args:
            orientations (numpy.ndarray): Array of orientation matrices
            
        Returns:
            tuple: (image, position array)
        """
        # Scale orientations by astar
        recips = orientations * self.params['astar']
        
        # Process each orientation
        for grain_nr, recip in enumerate(recips):
            recip = recip.reshape((3, 3))
            self.get_spots(recip, grain_nr)
        
        # Apply Gaussian filter to simulate detector response
        self.img = ndimg.gaussian_filter(self.img, self.params['gaussWidth']).astype(np.uint16)
        self.pos_arr = np.array(self.pos_arr)
        
        return self.img, self.pos_arr, recips
    
    def save_results(self, output_file, orientations):
        """
        Save simulation results to HDF5 and TIFF files.
        
        Args:
            output_file (str): Output file path
            orientations (numpy.ndarray): Original orientation matrices
        """
        # Save TIFF image
        tiff_path = f"{output_file}.tif"
        Image.fromarray(self.img).save(tiff_path)
        logger.info(f"Saved TIFF image to {tiff_path}")
        
        # Save HDF5 file with all data
        try:
            with h5py.File(output_file, 'w') as hf:
                hf.create_dataset('/entry1/data/data', data=self.img)
                
                # Reshape recips to store in file
                recips = orientations * self.params['astar']
                hf.create_dataset('/entry1/recips', data=recips)
                
                # Store spot positions
                hf.create_dataset('/entry1/spots', data=self.pos_arr)
                
                # Store original orientation matrices
                hf.create_dataset('/entry1/orientation_matrices', data=orientations)
                
            logger.info(f"Saved HDF5 data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving HDF5 file: {str(e)}")


def ensure_hkl_file_exists(params, install_path):
    """
    Ensure the HKL file exists, generating it if necessary.
    
    Args:
        params (dict): Configuration parameters
        install_path (str): Installation directory path
        
    Returns:
        numpy.ndarray: HKL reflection data
    """
    hkl_file = params['hklf']
    
    if not os.path.exists(hkl_file):
        logger.info(f"HKL file {hkl_file} not found, generating...")
        
        # Build command to generate HKLs
        python_path = sys.executable
        cmd = [
            f'{python_path}',
            f'{install_path}/GenerateHKLs.py',
            f'-resultFileName {hkl_file}',
            f'-sgnum {params["sgNum"]}',
            f'-sym {params["sym"]}',
            f'-latticeParameter {params["latC"]}',
            f'-RArray {params["r_arr"]}',
            f'-PArray {params["p_arr"]}',
            f'-NumPxX {params["nPxX"]}',
            f'-NumPxY {params["nPxY"]}',
            f'-dx {params["dx"]}',
            f'-dy {params["dy"]}'
        ]
        
        command = ' '.join(cmd)
        try:
            subprocess.run(command, shell=True, check=True)
            logger.info(f"Generated HKL file: {hkl_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating HKL file: {str(e)}")
            sys.exit(1)
    
    # Load HKL data
    try:
        hkl_data = np.genfromtxt(hkl_file)
        return hkl_data
    except Exception as e:
        logger.error(f"Error loading HKL file: {str(e)}")
        sys.exit(1)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    class CustomParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write(f'error: {message}\n')
            self.print_help()
            sys.exit(2)
    
    parser = CustomParser(
        description='LaueMatching Generate Simulation using an experiment configuration and a list of orientations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-configFile', 
        type=str, 
        required=True, 
        help='Configuration file to run the simulation.'
    )
    parser.add_argument(
        '-orientationFile', 
        type=str, 
        required=True, 
        help='File containing list of orientations to simulate. Must be orientation matrices with determinant 1. Each row has one orientation.'
    )
    parser.add_argument(
        '-outputFile', 
        type=str, 
        required=True, 
        help='File to save the data. Must be of the format FILESTEM_XX.h5, where XX is the file number, eg. 1.'
    )
    parser.add_argument(
        '-skipPercentage',
        type=float,
        default=30.0,
        help='Percentage of spots to randomly skip (0-100). Use 0 for no skipping.'
    )
    
    args, unparsed = parser.parse_known_args()
    return args


def main():
    """Main function to run the simulation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Get installation path (scripts/ directory, where GenerateHKLs.py lives)
    install_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Installation path: {install_path}")
    
    # Parse configuration file
    logger.info(f"Parsing configuration file: {args.configFile}")
    config_parser = ConfigParser(args.configFile)
    params = config_parser.get_params()
    
    # Load orientations
    logger.info(f"Loading orientations from: {args.orientationFile}")
    orientation_loader = OrientationLoader(args.orientationFile)
    orientations = orientation_loader.load_orientations()
    logger.info(f"Loaded {len(orientations)} orientations")
    
    # Ensure HKL file exists and load it
    hkl_data = ensure_hkl_file_exists(params, install_path)
    logger.info(f"Loaded {len(hkl_data)} HKL reflections")
    
    # Initialize simulator with the skip percentage
    logger.info(f"Setting spot skip percentage to: {args.skipPercentage}%")
    simulator = DiffractionSimulator(params, hkl_data, skip_percentage=args.skipPercentage)
    
    # Run simulation
    logger.info("Running simulation...")
    simulator.simulate(orientations)
    
    # Save results
    logger.info(f"Saving results to: {args.outputFile}")
    simulator.save_results(args.outputFile, orientations)
    
    logger.info("Simulation completed successfully")


if __name__ == "__main__":
    main()