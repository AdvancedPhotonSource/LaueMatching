/**
 * @file geometry.h
 * @brief Geometric utilities for Laue pattern matching
 * 
 * Provides functions for handling quaternions, rotation matrices,
 * Euler angles, and other geometric operations.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

#ifndef LAUE_GEOMETRY_H
#define LAUE_GEOMETRY_H

/**
 * @brief Convert between sin/cos values and angle
 * 
 * @param s Sine value
 * @param c Cosine value
 * @return Corresponding angle in radians
 */
double geometry_sin_cos_to_angle(double s, double c);

/**
 * @brief Normalize a quaternion
 * 
 * @param quat Quaternion to normalize (modified in-place)
 */
void geometry_normalize_quaternion(double quat[4]);

/**
 * @brief Compute quaternion product: Q = r * q
 * 
 * @param q First quaternion
 * @param r Second quaternion
 * @param Q Result of product
 */
void geometry_quaternion_product(const double q[4], const double r[4], double Q[4]);

/**
 * @brief Convert orientation matrix to quaternion
 * 
 * @param orientMat Orientation matrix (9 elements)
 * @param quat Resulting quaternion
 */
void geometry_orientation_matrix_to_quaternion(const double orientMat[9], double quat[4]);

/**
 * @brief Convert 3x3 orientation matrix to quaternion
 * 
 * @param orientMat Orientation matrix (3x3)
 * @param quat Resulting quaternion
 */
void geometry_orientation_matrix_3x3_to_quaternion(const double orientMat[3][3], double quat[4]);

/**
 * @brief Convert Euler angles to orientation matrix
 * 
 * @param euler Euler angles (radians)
 * @param mat Resulting 3x3 orientation matrix
 */
void geometry_euler_to_orientation_matrix(const double euler[3], double mat[3][3]);

/**
 * @brief Convert orientation matrix to Euler angles
 * 
 * @param mat 3x3 orientation matrix
 * @param euler Resulting Euler angles (radians)
 */
void geometry_orientation_matrix_to_euler(const double mat[3][3], double euler[3]);

/**
 * @brief Multiply 3x3 matrices: res = m * n
 * 
 * @param m First 3x3 matrix
 * @param n Second 3x3 matrix
 * @param res Resulting 3x3 matrix
 */
void geometry_matrix_multiply_3x3(const double m[3][3], const double n[3][3], double res[3][3]);

/**
 * @brief Multiply 3x3 matrix with 3D vector: r = m * v
 * 
 * @param m 3x3 matrix
 * @param v 3D vector
 * @param r Resulting 3D vector
 */
void geometry_matrix_vector_multiply(const double m[3][3], const double v[3], double r[3]);

/**
 * @brief Zero out very small values to improve numerical stability
 * 
 * @param val Value to check
 * @return Same value or 0 if below threshold
 */
double geometry_zero_out(double val);

/**
 * @brief Create rotation matrix from axis-angle representation
 * 
 * @param axis 3D axis vector
 * @param angle Rotation angle in radians
 * @param matrix Resulting 3x3 rotation matrix
 */
void geometry_axis_angle_to_matrix(const double axis[3], double angle, double matrix[3][3]);

/**
 * @brief Create rotation matrix transpose from axis-angle representation
 * 
 * @param axis 3D axis vector
 * @param angle Rotation angle in radians
 * @param matrix Resulting 3x3 transposed rotation matrix
 */
void geometry_axis_angle_to_matrix_transpose(const double axis[3], double angle, double matrix[3][3]);

/**
 * @brief Get misorientation angle between two orientations represented as quaternions
 * 
 * @param quat1 First quaternion
 * @param quat2 Second quaternion
 * @return Misorientation angle in degrees
 */
double geometry_get_misorientation(const double quat1[4], const double quat2[4]);

/**
 * @brief Bring quaternion to fundamental region based on symmetry
 * 
 * @param quatIn Input quaternion
 * @param quatOut Output quaternion in fundamental region
 * @param symm Symmetry operators array
 * @param nSym Number of symmetry operators
 */
void geometry_bring_to_fundamental_region(
    const double quatIn[4], 
    double quatOut[4], 
    const double symm[][4], 
    int nSym
);

#endif /* LAUE_GEOMETRY_H */