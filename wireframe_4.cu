#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int width = 800;
const int height = 800;

struct Vec3f {
    float x, y, z;
    __host__ __device__ Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    __host__ __device__ Vec3f operator+(const Vec3f& v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3f operator-(const Vec3f& v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }

    __host__ __device__ Vec3f rotate(float pitch, float yaw, float roll) const {
        float cy = cosf(yaw);
        float sy = sinf(yaw);
        Vec3f yawed(cy * x + sy * z, y, -sy * x + cy * z);

        float cp = cosf(pitch);
        float sp = sinf(pitch);
        Vec3f pitched(yawed.x, cp * yawed.y - sp * yawed.z, sp * yawed.y + cp * yawed.z);

        float cr = cosf(roll);
        float sr = sinf(roll);
        return Vec3f(cr * pitched.x - sr * pitched.y, sr * pitched.x + cr * pitched.y, pitched.z);
    }
};

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ Vec3f perspectiveProject(const Vec3f& v, float fov, float aspect, float near, float far) {
    float fovRad = fov * M_PI / 180.0f;
    float tanHalfFov = tanf(fovRad / 2.0f);
    
    float x = v.x / (v.z * tanHalfFov);
    float y = v.y / (v.z * tanHalfFov * aspect);
    float z = (v.z - near) / (far - near);
    
    return Vec3f(x, y, z);
}

__device__ void drawLine(int x0, int y0, int x1, int y1, unsigned char* image, int width, int height) {
    bool steep = false;
    if (abs(x0 - x1) < abs(y0 - y1)) {
        swap(x0, y0);
        swap(x1, y1);
        steep = true;
    }
    if (x0 > x1) {
        swap(x0, x1);
        swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    int derror2 = abs(dy) * 2;
    int error2 = 0;
    int y = y0;

    for (int x = x0; x <= x1; x++) {
        if (steep) {
            if (y >= 0 && y < width && x >= 0 && x < height) {
                int index = (x * width + y) * 3;
                image[index] = image[index + 1] = image[index + 2] = 255;
            }
        } else {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                int index = (y * width + x) * 3;
                image[index] = image[index + 1] = image[index + 2] = 255;
            }
        }
        error2 += derror2;
        if (error2 > dx) {
            y += (y1 > y0 ? 1 : -1);
            error2 -= dx * 2;
        }
    }
}

__global__ void drawWireframeKernel(Vec3f* verts, int* faces, int numFaces, unsigned char* image,
                                    Vec3f position, float pitch, float yaw, float roll,
                                    float fov, float aspect, float near, float far, int width, int height) {
    extern __shared__ int sharedFaces[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFaces) return;

    // Load face indices into shared memory
    if (threadIdx.x < blockDim.x) {
        for (int j = 0; j < 3; j++) {
            sharedFaces[threadIdx.x * 3 + j] = faces[idx * 3 + j];
        }
    }
    __syncthreads();

    for (int j = 0; j < 3; j++) {
        Vec3f v0 = verts[sharedFaces[threadIdx.x * 3 + j]].rotate(pitch, yaw, roll) - position;
        Vec3f v1 = verts[sharedFaces[threadIdx.x * 3 + (j + 1) % 3]].rotate(pitch, yaw, roll) - position;

        v0 = perspectiveProject(v0, fov, aspect, near, far);
        v1 = perspectiveProject(v1, fov, aspect, near, far);

        int x0 = (v0.x + 1.0f) * width / 2.0f;
        int y0 = (v0.y + 1.0f) * height / 2.0f;
        int x1 = (v1.x + 1.0f) * width / 2.0f;
        int y1 = (v1.y + 1.0f) * height / 2.0f;

        drawLine(x0, y0, x1, y1, image, width, height);
    }
}

bool loadObj(const std::string& filename, std::vector<Vec3f>& verts, std::vector<int>& faces) {
    std::ifstream inFile(filename);
    if (inFile.fail()) return false;

    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        char prefix;

        if (line.substr(0, 2) == "v ") {
            Vec3f vec;
            iss >> prefix >> vec.x >> vec.y >> vec.z;
            verts.push_back(vec);
        } else if (line.substr(0, 2) == "f ") {
            std::vector<int> face;
            int vIndex;
            iss >> prefix;
            while (iss >> vIndex) {
                face.push_back(vIndex - 1);
                iss.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
            }
            faces.insert(faces.end(), face.begin(), face.end());
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<Vec3f> vertices;
    std::vector<int> faces;

    if (!loadObj("african_head.obj", vertices, faces)) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return 1;
    }

    Vec3f position(0, 0, 3);
    float fov = 60.0f;
    float aspect = (float)width / (float)height;
    float near = 0.1f;
    float far = 100.0f;
    float pitch = 0.0f;
    float yaw = M_PI / 4;
    float roll = 0.0f;

    Vec3f* d_verts;
    int* d_faces;
    unsigned char* d_image;

    cudaMalloc(&d_verts, vertices.size() * sizeof(Vec3f));
    cudaMalloc(&d_faces, faces.size() * sizeof(int));
    cudaMalloc(&d_image, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_verts, vertices.data(), vertices.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_image, 0, width * height * 3 * sizeof(unsigned char));

    int numFaces = faces.size() / 3;
    int threadsPerBlock = 256;
    int numBlocks = (numFaces + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * 3 * sizeof(int);

    drawWireframeKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_verts, d_faces, numFaces, d_image,
                                                        position, pitch, yaw, roll,
                                                        fov, aspect, near, far, width, height);
    cudaDeviceSynchronize();

    unsigned char* h_image = new unsigned char[width * height * 3];
    cudaMemcpy(h_image, d_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", width, height, 3, h_image, width * 3);

    delete[] h_image;
    cudaFree(d_verts);
    cudaFree(d_faces);
    cudaFree(d_image);

    return 0;
}