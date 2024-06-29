#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

// --- Vector and Matrix structs ---
struct Vec3f {
    float x, y, z;

    __host__ __device__ Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3f operator+(const Vec3f& v) const {
        return Vec3f(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3f operator-(const Vec3f& v) const {
        return Vec3f(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3f operator*(float f) const {
        return Vec3f(x * f, y * f, z * f);
    }

    __host__ __device__ Vec3f operator/(float f) const {
        return Vec3f(x / f, y / f, z / f);
    }

    __host__ __device__ float dot(const Vec3f& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ Vec3f cross(const Vec3f& v) const {
        return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    __host__ __device__ Vec3f normalize() const {
        float l = sqrt(x * x + y * y + z * z);
        return Vec3f(x / l, y / l, z / l);
    }

    __host__ __device__ float& operator[](int i) {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    __host__ __device__ const float& operator[](int i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }
};

struct Vec2f {
    float u, v;

    __host__ __device__ Vec2f(float u = 0, float v = 0) : u(u), v(v) {}

    __host__ __device__ Vec2f operator*(float f) const {
        return Vec2f(u * f, v * f);
    }

    __host__ __device__ Vec2f operator+(const Vec2f& other) const {
        return Vec2f(u + other.u, v + other.v);
    }
};

struct Mat4f {
    float m[4][4];

    __host__ __device__ Mat4f() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    __host__ __device__ Vec3f transformPoint(const Vec3f& v) const {
        float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3];
        float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3];
        float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3];
        float w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
        return Vec3f(x / w, y / w, z / w);
    }

    __host__ __device__ Vec3f transformDirection(const Vec3f& v) const {
        float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
        float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
        float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;
        return Vec3f(x, y, z);
    }
};

// --- Triangle struct and barycentric calculation ---
struct Triangle {
    Vec3f vertices[3];
    Vec2f uvs[3];
    Vec3f normals[3];
};

__device__ Vec3f calculateBarycentricCoords(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f s[2] = {Vec3f(C.x - A.x, B.x - A.x, A.x - P.x),
                  Vec3f(C.y - A.y, B.y - A.y, A.y - P.y)};
    Vec3f u = s[0].cross(s[1]);
    return std::abs(u.z) > 1e-2 ? Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z,
                                     u.x / u.z)
                               : Vec3f(-1, 1, 1);
}

// --- Perspective matrix generation ---
__host__ __device__ Mat4f createPerspectiveMatrix(float fov, float aspect, float near,
                                                  float far) {
    Mat4f result;
    float tanHalfFov = tan(fov / 2.0f);
    result.m[0][0] = 1.0f / (aspect * tanHalfFov);
    result.m[1][1] = 1.0f / tanHalfFov;
    result.m[2][2] = -(far + near) / (far - near);
    result.m[2][3] = -2.0f * far * near / (far - near);
    result.m[3][2] = -1.0f;
    return result;
}

// --- Kernel function ---
__global__ void rasterize(const Vec3f* projectedVertices, const Triangle* triangles,
                         const int* triangleCounts, const unsigned char* textures,
                         const int* texWidths, const int* texHeights,
                         const Mat4f* modelMatrices, unsigned char* output,
                         float* zbuffer, int width, int height, int numObjects) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int flippedY = height - 1 - y;
    int pixelIndex = flippedY * width + x;
    zbuffer[pixelIndex] = FLT_MAX;

    Vec3f color(0.2f, 0.2f, 0.2f);
    Vec3f lightDir = Vec3f(1, 1, 1).normalize();
    Vec3f P(x, flippedY, 0);

    int vertexOffset = 0;
    int triangleOffset = 0;
    int textureOffset = 0;

    for (int obj = 0; obj < numObjects; ++obj) {
        for (int i = 0; i < triangleCounts[obj]; ++i) {
            const Triangle& tri = triangles[triangleOffset + i];
            Vec3f screenCoords[3];
            for (int j = 0; j < 3; ++j) {
                screenCoords[j] =
                        Vec3f((projectedVertices[vertexOffset + i * 3 + j].x + 1.0f) *
                              width / 2.0f,
                              (1.0f - projectedVertices[vertexOffset + i * 3 + j].y) *
                              height / 2.0f,
                              projectedVertices[vertexOffset + i * 3 + j].z);
            }

            Vec3f bcScreen =
                    calculateBarycentricCoords(screenCoords[0], screenCoords[1],
                                            screenCoords[2], P);
            if (bcScreen.x < 0 || bcScreen.y < 0 || bcScreen.z < 0)
                continue;

            float fragDepth = bcScreen.x * screenCoords[0].z +
                             bcScreen.y * screenCoords[1].z +
                             bcScreen.z * screenCoords[2].z;
            if (fragDepth < zbuffer[pixelIndex]) {
                zbuffer[pixelIndex] = fragDepth;

                Vec2f uv = tri.uvs[0] * bcScreen.x + tri.uvs[1] * bcScreen.y +
                           tri.uvs[2] * bcScreen.z;
                int texX = uv.u * texWidths[obj];
                int texY = (1.0f - uv.v) * texHeights[obj];
                int texIndex = textureOffset + (texY * texWidths[obj] + texX) * 3;

                Vec3f texColor(textures[texIndex] / 255.0f,
                               textures[texIndex + 1] / 255.0f,
                               textures[texIndex + 2] / 255.0f);

                Vec3f normal = (tri.normals[0] * bcScreen.x +
                                tri.normals[1] * bcScreen.y +
                                tri.normals[2] * bcScreen.z)
                                       .normalize();
                normal = modelMatrices[obj].transformDirection(normal);

                float diffuse = fmaxf(0.0f, normal.dot(lightDir));
                color = texColor * (0.3f + 0.7f * diffuse); 

            }
        }
        vertexOffset += triangleCounts[obj] * 3;
        triangleOffset += triangleCounts[obj];
        textureOffset += texWidths[obj] * texHeights[obj] * 3;
    }


       output[pixelIndex * 3 + 0] =
               static_cast<unsigned char>(fminf(color.x * 255.0f, 255.0f));
       output[pixelIndex * 3 + 1] =
               static_cast<unsigned char>(fminf(color.y * 255.0f, 255.0f));
       output[pixelIndex * 3 + 2] =
               static_cast<unsigned char>(fminf(color.z * 255.0f, 255.0f));
}

// --- OBJ loading function ---
bool loadObjFile(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open OBJ file: %s\n", filename);
        return false;
    }

    std::vector<Vec3f> vertices;
    std::vector<Vec2f> texcoords;
    std::vector<Vec3f> normals;

    std::string line, type;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> type;
        if (type == "v") {
            Vec3f v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (type == "vt") {
            Vec2f vt;
            iss >> vt.u >> vt.v;
            texcoords.push_back(vt);
        } else if (type == "vn") {
            Vec3f vn;
            iss >> vn.x >> vn.y >> vn.z;
            normals.push_back(vn);
        } else if (type == "f") {
            Triangle tri;
            for (int i = 0; i < 3; ++i) {
                int v, vt, vn;
                char slash;
                iss >> v >> slash >> vt >> slash >> vn;
                tri.vertices[i] = vertices[v - 1];
                tri.uvs[i] = texcoords[vt - 1];
                tri.normals[i] = normals[vn - 1];
            }
            triangles.push_back(tri);
        }
    }
    return true;
}

// --- Main function ---
int main() {
    const int width = 800, height = 600;
    const int numObjects = 2;

    std::vector<Triangle> triangles[numObjects];
    unsigned char* textures[numObjects];
    int texWidths[numObjects], texHeights[numObjects];

    if (!loadObjFile("african_head.obj", triangles[0]) ||
        !loadObjFile("drone.obj", triangles[1])) {
        return 1;
    }

    textures[0] =
            stbi_load("african_head_diffuse.tga", &texWidths[0], &texHeights[0],
                      nullptr, 3);
    textures[1] = stbi_load("drone.png", &texWidths[1], &texHeights[1], nullptr,
                            3);

    for (int i = 0; i < numObjects; ++i) {
        if (!textures[i]) {
            fprintf(stderr, "Failed to load texture for object %d\n", i);
            return 1;
        }
    }

    Mat4f modelMatrices[numObjects] = {Mat4f(), Mat4f()};
    // First object transformation
    modelMatrices[0].m[0][3] = -1.0f;
    modelMatrices[0].m[2][3] = -3.0f;
    modelMatrices[0].m[0][0] = modelMatrices[0].m[2][2] =
            cos(3.14159f / 4.0f);
    modelMatrices[0].m[0][2] = -(modelMatrices[0].m[2][0] =
            -sin(3.14159f / 4.0f));
    // Second object transformation
    modelMatrices[1].m[0][3] = 1.0f;
    modelMatrices[1].m[1][3] = 0.5f;
    modelMatrices[1].m[2][3] = -2.5f;
    modelMatrices[1].m[0][0] = modelMatrices[1].m[1][1] =
    modelMatrices[1].m[2][2] = 0.1f;

    Mat4f proj =
            createPerspectiveMatrix(3.14159f / 4.0f, (float)width / height, 0.1f,
                                   100.0f);

    std::vector<Vec3f> projectedVertices;
    for (int i = 0; i < numObjects; ++i) {
        for (const auto& tri : triangles[i]) {
            for (int j = 0; j < 3; ++j) {
                projectedVertices.push_back(
                        proj.transformPoint(modelMatrices[i].transformPoint(tri.vertices[j])));
            }
        }
    }

    // --- Allocate device memory ---
    Triangle* dTriangles;
    unsigned char* dTextures;
    int* dTriangleCounts, * dTexWidths, * dTexHeights;
    Mat4f* dModelMatrices;
    Vec3f* dProjectedVertices;
    unsigned char* dOutput;
    float* dZbuffer;

    int totalTriangles = triangles[0].size() + triangles[1].size();
    int totalTextureSize = (texWidths[0] * texHeights[0] +
                           texWidths[1] * texHeights[1]) *
                          3;

    CHECK_CUDA(cudaMalloc(&dTriangles, totalTriangles * sizeof(Triangle)));
    CHECK_CUDA(
            cudaMalloc(&dTextures, totalTextureSize * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&dTriangleCounts, numObjects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dTexWidths, numObjects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dTexHeights, numObjects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dModelMatrices, numObjects * sizeof(Mat4f)));
    CHECK_CUDA(cudaMalloc(&dProjectedVertices,
                           projectedVertices.size() * sizeof(Vec3f)));
    CHECK_CUDA(
            cudaMalloc(&dOutput, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&dZbuffer, width * height * sizeof(float)));

    // --- Copy data to device ---
    int triangleOffset = 0;
    int textureOffset = 0;
    for (int i = 0; i < numObjects; ++i) {
        CHECK_CUDA(cudaMemcpy(dTriangles + triangleOffset, triangles[i].data(),
                              triangles[i].size() * sizeof(Triangle),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dTextures + textureOffset, textures[i],
                              texWidths[i] * texHeights[i] * 3 *
                              sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
        triangleOffset += triangles[i].size();
        textureOffset += texWidths[i] * texHeights[i] * 3;
    }

    int triangleCounts[numObjects] = {(int)triangles[0].size(),
                                     (int)triangles[1].size()};
    CHECK_CUDA(cudaMemcpy(dTriangleCounts, triangleCounts,
                          numObjects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dTexWidths, texWidths, numObjects * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dTexHeights, texHeights, numObjects * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dModelMatrices, modelMatrices,
                          numObjects * sizeof(Mat4f), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dProjectedVertices, projectedVertices.data(),
                          projectedVertices.size() * sizeof(Vec3f),
                          cudaMemcpyHostToDevice));

    // --- Kernel launch ---
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    rasterize<<<gridSize, blockSize>>>(
            dProjectedVertices, dTriangles, dTriangleCounts, dTextures, dTexWidths,
            dTexHeights, dModelMatrices, dOutput, dZbuffer, width, height,
            numObjects);

    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Copy result back to host and save ---
    unsigned char* output = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output, dOutput,
                          width * height * 3 * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

    stbi_write_png("output.png", width, height, 3, output, width * 3);

    // --- Free resources ---
    delete[] output;
    for (int i = 0; i < numObjects; ++i) {
        stbi_image_free(textures[i]);
    }
    cudaFree(dTriangles);
    cudaFree(dTextures);
    cudaFree(dTriangleCounts);
    cudaFree(dTexWidths);
    cudaFree(dTexHeights);
    cudaFree(dModelMatrices);
    cudaFree(dProjectedVertices);
    cudaFree(dOutput);
    cudaFree(dZbuffer);

    return 0;
}