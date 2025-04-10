/**
 * @file geometry.c
 * @brief Implementation of geometric utilities
 */

#include "../common.h"
#include "geometry.h"
#include "crystallography.h"

double geometry_sin_cos_to_angle(double s, double c) {
    return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

void geometry_normalize_quaternion(double quat[4]) {
    double norm = sqrt(quat[0]*quat[0] + quat[1]*quat[1] + 
                       quat[2]*quat[2] + quat[3]*quat[3]);
    
    if (norm < LAUE_EPSILON) {
        // Invalid quaternion, set to identity
        quat[0] = 1.0;
        quat[1] = 0.0;
        quat[2] = 0.0;
        quat[3] = 0.0;
        return;
    }
    
    quat[0] /= norm;
    quat[1] /= norm;
    quat[2] /= norm;
    quat[3] /= norm;
}

void geometry_quaternion_product(const double q[4], const double r[4], double Q[4]) {
    Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3];
    Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3];
    Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1];
    Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2];
    
    // Ensure positive scalar component
    if (Q[0] < 0) {
        Q[0] = -Q[0];
        Q[1] = -Q[1];
        Q[2] = -Q[2];
        Q[3] = -Q[3];
    }
    
    geometry_normalize_quaternion(Q);
}

void geometry_orientation_matrix_to_quaternion(const double orientMat[9], double quat[4]) {
    double trace = orientMat[0] + orientMat[4] + orientMat[8];
    
    if (trace > 0) {
        double s = 0.5 / sqrt(trace + 1.0);
        quat[0] = 0.25 / s;
        quat[1] = (orientMat[7] - orientMat[5]) * s;
        quat[2] = (orientMat[2] - orientMat[6]) * s;
        quat[3] = (orientMat[3] - orientMat[1]) * s;
    } else {
        if (orientMat[0] > orientMat[4] && orientMat[0] > orientMat[8]) {
            double s = 2.0 * sqrt(1.0 + orientMat[0] - orientMat[4] - orientMat[8]);
            quat[0] = (orientMat[7] - orientMat[5]) / s;
            quat[1] = 0.25 * s;
            quat[2] = (orientMat[1] + orientMat[3]) / s;
            quat[3] = (orientMat[2] + orientMat[6]) / s;
        } else if (orientMat[4] > orientMat[8]) {
            double s = 2.0 * sqrt(1.0 + orientMat[4] - orientMat[0] - orientMat[8]);
            quat[0] = (orientMat[2] - orientMat[6]) / s;
            quat[1] = (orientMat[1] + orientMat[3]) / s;
            quat[2] = 0.25 * s;
            quat[3] = (orientMat[5] + orientMat[7]) / s;
        } else {
            double s = 2.0 * sqrt(1.0 + orientMat[8] - orientMat[0] - orientMat[4]);
            quat[0] = (orientMat[3] - orientMat[1]) / s;
            quat[1] = (orientMat[2] + orientMat[6]) / s;
            quat[2] = (orientMat[5] + orientMat[7]) / s;
            quat[3] = 0.25 * s;
        }
    }
    
    // Ensure positive scalar component
    if (quat[0] < 0) {
        quat[0] = -quat[0];
        quat[1] = -quat[1];
        quat[2] = -quat[2];
        quat[3] = -quat[3];
    }
    
    geometry_normalize_quaternion(quat);
}

void geometry_orientation_matrix_3x3_to_quaternion(const double orientMat[3][3], double quat[4]) {
    double flatMat[9];
    int i, j, idx = 0;
    
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            flatMat[idx++] = orientMat[i][j];
        }
    }
    
    geometry_orientation_matrix_to_quaternion(flatMat, quat);
}

void geometry_euler_to_orientation_matrix(const double euler[3], double m_out[3][3]) {
    double psi = euler[0];
    double phi = euler[1];
    double theta = euler[2];
    
    double cps = cos(psi);
    double cph = cos(phi);
    double cth = cos(theta);
    double sps = sin(psi);
    double sph = sin(phi);
    double sth = sin(theta);
    
    m_out[0][0] = cth * cps - sth * cph * sps;
    m_out[0][1] = -cth * cph * sps - sth * cps;
    m_out[0][2] = sph * sps;
    m_out[1][0] = cth * sps + sth * cph * cps;
    m_out[1][1] = cth * cph * cps - sth * sps;
    m_out[1][2] = -sph * cps;
    m_out[2][0] = sth * sph;
    m_out[2][1] = cth * sph;
    m_out[2][2] = cph;
}

void geometry_orientation_matrix_to_euler(const double m[3][3], double euler[3]) {
    double psi, phi, theta, sph;

    if (fabs(m[2][2] - 1.0) < LAUE_EPSILON) {
        phi = 0;
    } else {
        phi = acos(m[2][2]);
    }
    
    sph = sin(phi);
    
    if (fabs(sph) < LAUE_EPSILON) {
        psi = 0.0;
        theta = (fabs(m[2][2] - 1.0) < LAUE_EPSILON) ? 
                geometry_sin_cos_to_angle(m[1][0], m[0][0]) : 
                geometry_sin_cos_to_angle(-m[1][0], m[0][0]);
    } else {
        psi = (fabs(-m[1][2] / sph) <= 1.0) ? 
              geometry_sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph) : 
              geometry_sin_cos_to_angle(m[0][2] / sph, 1);
              
        theta = (fabs(m[2][1] / sph) <= 1.0) ? 
                geometry_sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph) : 
                geometry_sin_cos_to_angle(m[2][0] / sph, 1);
    }
    
    euler[0] = psi;
    euler[1] = phi;
    euler[2] = theta;
}

void geometry_matrix_multiply_3x3(const double m[3][3], const double n[3][3], double res[3][3]) {
    int r, c, i;
    
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            res[r][c] = 0;
            for (i = 0; i < 3; i++) {
                res[r][c] += m[r][i] * n[i][c];
            }
        }
    }
}

void geometry_matrix_vector_multiply(const double m[3][3], const double v[3], double r[3]) {
    int i, j;
    
    r[0] = 0;
    r[1] = 0;
    r[2] = 0;
    
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            r[i] += m[i][j] * v[j];
        }
    }
}

double geometry_zero_out(double val) {
    return (fabs(val) < LAUE_EPSILON) ? 0 : val;
}

void geometry_axis_angle_to_matrix(const double axis[3], double angle, double matrix[3][3]) {
    double c = cos(angle);
    double s = sin(angle);
    double t = 1.0 - c;
    double x = axis[0];
    double y = axis[1];
    double z = axis[2];
    
    matrix[0][0] = c + x*x*t;
    matrix[1][1] = c + y*y*t;
    matrix[2][2] = c + z*z*t;
    
    double tmp1 = x*y*t;
    double tmp2 = z*s;
    matrix[0][1] = tmp1 - tmp2;
    matrix[1][0] = tmp1 + tmp2;
    
    tmp1 = x*z*t;
    tmp2 = y*s;
    matrix[0][2] = tmp1 + tmp2;
    matrix[2][0] = tmp1 - tmp2;
    
    tmp1 = y*z*t;
    tmp2 = x*s;
    matrix[1][2] = tmp1 - tmp2;
    matrix[2][1] = tmp1 + tmp2;
}

void geometry_axis_angle_to_matrix_transpose(const double axis[3], double angle, double matrix[3][3]) {
    geometry_axis_angle_to_matrix(axis, angle, matrix);
    
    // Transpose the matrix
    double temp;
    temp = matrix[0][1]; matrix[0][1] = matrix[1][0]; matrix[1][0] = temp;
    temp = matrix[0][2]; matrix[0][2] = matrix[2][0]; matrix[2][0] = temp;
    temp = matrix[1][2]; matrix[1][2] = matrix[2][1]; matrix[2][1] = temp;
}

double geometry_get_misorientation(const double quat1[4], const double quat2[4]) {
    // Get symmetries
    extern int crystal_symmetry_count;
    extern double crystal_symmetry_operators[][4];
    
    double q1FR[4], q2FR[4], q1Inv[4], QP[4], MisV[4];
    
    geometry_bring_to_fundamental_region(quat1, q1FR, crystal_symmetry_operators, crystal_symmetry_count);
    geometry_bring_to_fundamental_region(quat2, q2FR, crystal_symmetry_operators, crystal_symmetry_count);
    
    // Inverse of first quaternion
    q1Inv[0] = -q1FR[0];
    q1Inv[1] = q1FR[1];
    q1Inv[2] = q1FR[2];
    q1Inv[3] = q1FR[3];
    
    geometry_quaternion_product(q1Inv, q2FR, QP);
    geometry_bring_to_fundamental_region(QP, MisV, crystal_symmetry_operators, crystal_symmetry_count);
    
    if (MisV[0] > 1) MisV[0] = 1;
    double angle = 2 * (acos(MisV[0])) * LAUE_RAD2DEG;
    
    return angle;
}

void geometry_bring_to_fundamental_region(
    const double quatIn[4], 
    double quatOut[4], 
    const double symm[][4], 
    int nSym
) {
    int i;
    double qps[24][4]; // Assuming max symmetry operators is 24
    double q2[4], qt[4];
    double maxCos = -10000;
    int maxCosRowNr = 0;
    
    for (i = 0; i < nSym; i++) {
        q2[0] = symm[i][0];
        q2[1] = symm[i][1];
        q2[2] = symm[i][2];
        q2[3] = symm[i][3];
        
        geometry_quaternion_product(quatIn, q2, qt);
        
        qps[i][0] = qt[0];
        qps[i][1] = qt[1];
        qps[i][2] = qt[2];
        qps[i][3] = qt[3];
        
        if (maxCos < qt[0]) {
            maxCos = qt[0];
            maxCosRowNr = i;
        }
    }
    
    quatOut[0] = qps[maxCosRowNr][0];
    quatOut[1] = qps[maxCosRowNr][1];
    quatOut[2] = qps[maxCosRowNr][2];
    quatOut[3] = qps[maxCosRowNr][3];
    
    geometry_normalize_quaternion(quatOut);
}