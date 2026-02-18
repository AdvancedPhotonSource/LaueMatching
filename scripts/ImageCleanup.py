#!/usr/bin/env python3
"""
LaueMatching Image Cleanup.

This script processes diffraction images by subtracting background
and identifying features based on connected component analysis.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import sys
import os
import scipy.ndimage as ndimg
from skimage.measure import label, regionprops
import argparse
import diplib as dip
import matplotlib.pyplot as plt
from PIL import Image
import time


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom argument parser with improved error handling."""
    def error(self, message):
        sys.stderr.write(f'Error: {message}\n')
        self.print_help()
        sys.exit(2)


def apply_median_filter(image, radius):
    """
    Apply a median filter to an image.
    
    Args:
        image: Input image (dip.Image)
        radius: Radius for the median filter
        
    Returns:
        Filtered image
    """
    return dip.MedianFilter(image, radius)


def load_image(filename, image_type):
    """
    Load image from file.
    
    Args:
        filename: Path to the image file
        image_type: Type of image ('hdf', 'tiff', or 'jpg')
        
    Returns:
        Tuple of (image_data, output_filename)
    """
    try:
        if image_type == 'hdf':
            output_filename = f"{filename.split('.')[0]}_cleaned.{filename.split('.')[1]}"
            with h5py.File(filename, 'r') as hf:
                image_data = np.array(hf['entry1/data/data'][()]).astype(np.uint16)
        else:
            output_filename = f"{filename.split('.')[0]}_cleaned.{filename.split('.')[1]}.h5"
            image = Image.open(filename)
            image_data = (np.array(image).astype(np.uint16)[:,:,0] * (-1) + 255)
            image_data *= int(16000/255)
        
        return image_data, output_filename
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)


def compute_background(image, radius, num_passes):
    """
    Compute background by applying median filter multiple times.
    
    Args:
        image: Input image data
        radius: Radius for median filter
        num_passes: Number of median filter passes
        
    Returns:
        Background image
    """
    start_time = time.time()
    print("Computing background...")
    
    data = dip.Image(image)
    for i in range(num_passes):
        data = apply_median_filter(data, radius)
        percent_complete = (i + 1) / num_passes * 100
        print(f"  Pass {i+1}/{num_passes} complete ({percent_complete:.1f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"Background computation completed in {elapsed_time:.2f} seconds.")
    
    return data


def subtract_background_and_threshold(image, background, min_area, max_area):
    """
    Subtract background, apply threshold, and filter regions by area.
    
    Args:
        image: Original image data
        background: Computed background image
        min_area: Minimum area for connected components
        max_area: Maximum area for connected components
        
    Returns:
        Tuple of (corrected_image, labels, threshold_value)
    """
    # Subtract background
    corrected_image = image.astype(np.double) - background
    
    # Calculate dynamic threshold based on standard deviation
    thresh = 60 * (1 + np.std(corrected_image) // 60)
    print(f'Computed threshold: {thresh}')
    
    # Apply threshold
    corrected_image[corrected_image < thresh] = 0
    corrected_image = corrected_image.astype(np.uint16)
    
    # Label connected components
    labels, num_labels = ndimg.label(corrected_image)
    
    # Filter by area
    filtered_count = 0
    for label_num in range(1, num_labels + 1):
        component_size = np.sum(labels == label_num)
        if component_size < min_area or component_size > max_area:
            corrected_image[labels == label_num] = 0
            labels[labels == label_num] = 0
            filtered_count += 1
    
    print(f"Filtered out {filtered_count} regions based on size constraints")
    print(f"Remaining regions: {num_labels - filtered_count}")
    
    return corrected_image, labels, thresh


def visualize_results(image, labels, output_prefix, save_plot=True, show_plot=True):
    """
    Visualize processing results with centroids marked.
    
    Args:
        image: Processed image
        labels: Labeled components
        output_prefix: Prefix for output filename
        save_plot: Whether to save the plot to a file
        show_plot: Whether to display the plot
    """
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Use log scale for better visualization
    ax.imshow(np.log(image + 1), cmap='gray_r')
    
    # Plot centroids
    props = regionprops(labels)
    for prop in props:
        centroid = prop.centroid
        ax.plot(centroid[1], centroid[0], 'ks', 
                markerfacecolor='none', ms=10, 
                markeredgecolor='black', markeredgewidth=1)
    
    ax.axis('equal')
    
    if save_plot:
        plot_filename = f"{output_prefix}_visualization.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {plot_filename}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)


def save_results(output_filename, corrected_image, original_image, background, threshold):
    """
    Save processing results to HDF5 file.
    
    Args:
        output_filename: Output file path
        corrected_image: Processed image
        original_image: Original image
        background: Computed background
        threshold: Threshold value used
    """
    try:
        with h5py.File(output_filename, 'w') as hf:
            hf.create_dataset('/entry1/data/data', data=corrected_image)
            hf.create_dataset('/entry1/data/rawdata', data=original_image)
            hf.create_dataset('/entry1/data/background', data=background)
            hf.create_dataset('/entry1/data/threshold', data=threshold)
        print(f"Results saved to {output_filename}")
    except Exception as e:
        print(f"Error saving results: {e}")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = CustomArgumentParser(
        description='LaueMatching Image Cleanup.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-inputFile', type=str, required=True, 
                        help='Name of input h5 file in the dataExchange format.')
    parser.add_argument('-minArea', type=int, required=False, default=10, 
                        help='Minimum area of connected pixels qualifying signal.')
    parser.add_argument('-maxArea', type=int, required=False, default=1000, 
                        help='Maximum area of connected pixels qualifying signal.')
    parser.add_argument('-radius', type=int, required=False, default=101, 
                        help='Radius of median filter to apply to compute background.')
    parser.add_argument('-nPasses', type=int, required=False, default=5, 
                        help='Number of median filter passes to compute background.')
    parser.add_argument('-imageType', type=str, required=False, default='hdf', 
                        help='Image type option: hdf, tiff or jpg.')
    parser.add_argument('-savePlot', action='store_true',
                        help='Save visualization plot to file.')
    parser.add_argument('-noShowPlot', action='store_true',
                        help='Do not display visualization plot.')
    
    args, unparsed = parser.parse_known_args()
    
    # Validate arguments
    if args.minArea < 1:
        parser.error("minArea must be at least 1")
    if args.maxArea <= args.minArea:
        parser.error("maxArea must be greater than minArea")
    if args.radius < 1:
        parser.error("radius must be at least 1")
    if args.nPasses < 1:
        parser.error("nPasses must be at least 1")
    if args.imageType not in ['hdf', 'tiff', 'jpg']:
        parser.error("imageType must be one of: hdf, tiff, jpg")
    
    return args


def main():
    """Main function to orchestrate the image processing."""
    # Start timing
    total_start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"Processing file: {args.inputFile}")
    
    # Load image
    original_image, output_filename = load_image(args.inputFile, args.imageType)
    print(f"Image loaded: {original_image.shape}")
    
    # Compute background
    background = compute_background(original_image, args.radius, args.nPasses)
    
    # Process image
    corrected_image, labels, threshold = subtract_background_and_threshold(
        original_image, background, args.minArea, args.maxArea
    )
    
    # Visualize results
    output_prefix = os.path.splitext(output_filename)[0]
    visualize_results(
        corrected_image, 
        labels, 
        output_prefix,
        save_plot=args.savePlot,
        show_plot=not args.noShowPlot
    )
    
    # Save results
    save_results(
        output_filename, 
        corrected_image, 
        original_image, 
        background, 
        threshold
    )
    
    # Report total execution time
    total_elapsed_time = time.time() - total_start_time
    print(f"Total processing time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()