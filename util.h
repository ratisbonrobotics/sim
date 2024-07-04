#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <string>
#include <algorithm>
#pragma nv_diag_suppress 611
#include <opencv2/opencv.hpp>
#pragma nv_diag_default 611
#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma nv_diag_default 550
#include <cfloat>
#include <cmath>

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    __host__ __device__ Vec3 normalize() const { float l = sqrt(x * x + y * y + z * z); return Vec3(x / l, y / l, z / l); }
};

struct Vec2 {
    float u, v;
    __host__ __device__ Vec2() : u(0), v(0) {}
    __host__ __device__ Vec2(float u, float v) : u(u), v(v) {}
    __host__ __device__ Vec2 operator*(float f) const { return Vec2(u * f, v * f); }
    __host__ __device__ Vec2 operator+(const Vec2& other) const { return Vec2(u + other.u, v + other.v); }
};

struct Mat4 {
    float m[16];

    __host__ __device__ Mat4() {
        for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }

    __host__ __device__ Mat4(float m00, float m01, float m02, float m03,
                             float m10, float m11, float m12, float m13,
                             float m20, float m21, float m22, float m23,
                             float m30, float m31, float m32, float m33) {
        m[0] = m00; m[1] = m01; m[2] = m02; m[3] = m03;
        m[4] = m10; m[5] = m11; m[6] = m12; m[7] = m13;
        m[8] = m20; m[9] = m21; m[10] = m22; m[11] = m23;
        m[12] = m30; m[13] = m31; m[14] = m32; m[15] = m33;
    }

    __host__ __device__ Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i*4 + j] = 0;
                for (int k = 0; k < 4; k++) {
                    result.m[i*4 + j] += m[i*4 + k] * other.m[k*4 + j];
                }
            }
        }
        return result;
    }

    __host__ __device__ Vec3 multiplyPoint(const Vec3& v) const {
        float x = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
        float y = m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7];
        float z = m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11];
        float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];
        return Vec3(x/w, y/w, z/w);
    }

    __host__ __device__ Vec3 multiplyVector(const Vec3& v) const {
        return Vec3(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z
        );
    }
};


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

struct Triangle {
    Vec3 v[3];
    Vec2 uv[3];
    Vec3 n[3];
};

Mat4 create_projection_matrix(float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    float f = 1.0f / tanHalfFov;
    float nf = 1.0f / (near - far);

    return Mat4(
        f / aspect,  0.0f,       0.0f,                   0.0f,
        0.0f,        f,          0.0f,                   0.0f,
        0.0f,        0.0f,       (far + near) * nf,      2.0f * far * near * nf,
        0.0f,        0.0f,       -1.0f,                  0.0f
    );
}

Mat4 create_model_matrix_random() {
    static std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> dis_pos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dis_scale(0.8f, 1.3f);
    std::uniform_real_distribution<float> dis_rot(0.0f, 2.0f * 3.14159f);

    float tx = dis_pos(gen);
    float ty = dis_pos(gen);
    float tz = dis_pos(gen) - 5.0f;  // Ensure object is in front of the camera
    float scale = dis_scale(gen);
    float rotation = dis_rot(gen);

    float cos_r = cos(rotation);
    float sin_r = sin(rotation);

    return Mat4(
        cos_r * scale,  0.0f,           -sin_r * scale,  tx,
        0.0f,           scale,          0.0f,            ty,
        sin_r * scale,  0.0f,           cos_r * scale,   tz,
        0.0f,           0.0f,           0.0f,            1.0f
    );
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Failed to open OBJ file: %s\n", filename);
        return;
    }

    std::vector<Vec3> vertices, normals;
    std::vector<Vec2> texcoords;
    std::string line, type;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> type;

        if (type == "v") {
            Vec3 v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (type == "vt") {
            Vec2 vt;
            iss >> vt.u >> vt.v;
            texcoords.push_back(vt);
        } else if (type == "vn") {
            Vec3 vn;
            iss >> vn.x >> vn.y >> vn.z;
            normals.push_back(vn);
        } else if (type == "f") {
            Triangle tri;
            for (int i = 0; i < 3; i++) {
                int v, vt, vn;
                char slash;
                iss >> v >> slash >> vt >> slash >> vn;
                tri.v[i] = vertices[v - 1];
                tri.uv[i] = texcoords[vt - 1];
                tri.n[i] = normals[vn - 1];
            }
            triangles.push_back(tri);
        }
    }
}
