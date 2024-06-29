#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int WIDTH = 800;
const int HEIGHT = 800;
const int BLOCK_SIZE = 256;

struct Vec3f {
    float x, y, z;
    __host__ __device__ Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3f operator-(const Vec3f& v) const {
        return Vec3f(x - v.x, y - v.y, z - v.z);
    }
};

struct Vertices {
    float* x;
    float* y;
    float* z;
};

__device__ int abs(int v) {
    return v < 0 ? -v : v;
}

__device__ Vec3f rotate(const Vec3f& v, float pitch, float yaw, float roll) {
    float cp = cosf(pitch), sp = sinf(pitch);
    float cy = cosf(yaw), sy = sinf(yaw);
    float cr = cosf(roll), sr = sinf(roll);

    Vec3f result;
    result.x = v.x * (cy * cr + sy * sp * sr) - v.y * (cy * sr - sy * sp * cr) + v.z * (sy * cp);
    result.y = v.x * (cp * sr) + v.y * (cp * cr) - v.z * sp;
    result.z = v.x * (sy * cr - cy * sp * sr) - v.y * (sy * sr + cy * sp * cr) + v.z * (cy * cp);
    return result;
}

__device__ Vec3f perspectiveProject(const Vec3f& v, float fov, float aspect, float near, float far) {
    float fovRad = fov * M_PI / 180.0f;
    float tanHalfFov = tanf(fovRad / 2.0f);
    
    Vec3f projected;
    projected.x = v.x / (v.z * tanHalfFov);
    projected.y = v.y / (v.z * tanHalfFov * aspect);
    projected.z = (v.z - near) / (far - near);
    
    return projected;
}

__device__ void drawLine(int x0, int y0, int x1, int y1, unsigned char* image) {
    bool steep = false;
    if (abs(x0 - x1) < abs(y0 - y1)) {
        int temp = x0; x0 = y0; y0 = temp;
        temp = x1; x1 = y1; y1 = temp;
        steep = true;
    }
    if (x0 > x1) {
        int temp = x0; x0 = x1; x1 = temp;
        temp = y0; y0 = y1; y1 = temp;
    }
    int dx = x1 - x0;
    int dy = abs(y1 - y0);
    int error = dx / 2;
    int ystep = (y0 < y1) ? 1 : -1;
    int y = y0;

    for (int x = x0; x <= x1; x++) {
        if (steep) {
            if (y >= 0 && y < WIDTH && x >= 0 && x < HEIGHT) {
                int index = (x * WIDTH + y) * 3;
                image[index] = image[index + 1] = image[index + 2] = 255;
            }
        } else {
            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                int index = (y * WIDTH + x) * 3;
                image[index] = image[index + 1] = image[index + 2] = 255;
            }
        }
        error -= dy;
        if (error < 0) {
            y += ystep;
            error += dx;
        }
    }
}

__global__ void drawWireframeKernel(Vertices verts, int* faces, int numFaces, unsigned char* image,
                                    Vec3f position, float pitch, float yaw, float roll,
                                    float fov, float aspect, float near, float far) {
    __shared__ Vec3f sharedVerts[BLOCK_SIZE * 3];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFaces) return;

    // Load vertices for this face into shared memory
    for (int j = 0; j < 3; j++) {
        int vertIdx = faces[idx * 3 + j];
        sharedVerts[threadIdx.x * 3 + j] = Vec3f(verts.x[vertIdx], verts.y[vertIdx], verts.z[vertIdx]);
    }
    __syncthreads();

    for (int j = 0; j < 3; j++) {
        Vec3f v0 = sharedVerts[threadIdx.x * 3 + j];
        Vec3f v1 = sharedVerts[threadIdx.x * 3 + (j + 1) % 3];

        v0 = rotate(v0, pitch, yaw, roll);
        v1 = rotate(v1, pitch, yaw, roll);

        v0 = v0 - position;
        v1 = v1 - position;

        v0 = perspectiveProject(v0, fov, aspect, near, far);
        v1 = perspectiveProject(v1, fov, aspect, near, far);

        int x0 = (v0.x + 1.0f) * WIDTH / 2.0f;
        int y0 = (v0.y + 1.0f) * HEIGHT / 2.0f;
        int x1 = (v1.x + 1.0f) * WIDTH / 2.0f;
        int y1 = (v1.y + 1.0f) * HEIGHT / 2.0f;

        drawLine(x0, y0, x1, y1, image);
    }
}

bool loadObj(const std::string& filename, std::vector<float>& vertX, std::vector<float>& vertY, std::vector<float>& vertZ, std::vector<int>& faces) {
    std::ifstream inFile(filename);
    if (inFile.fail()) return false;

    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        char prefix;

        if (line.substr(0, 2) == "v ") {
            float x, y, z;
            iss >> prefix >> x >> y >> z;
            vertX.push_back(x);
            vertY.push_back(y);
            vertZ.push_back(z);
        } else if (line.substr(0, 2) == "f ") {
            int vIndex;
            iss >> prefix;
            while (iss >> vIndex) {
                faces.push_back(vIndex - 1);
                iss.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<float> vertX, vertY, vertZ;
    std::vector<int> faces;

    if (!loadObj("african_head.obj", vertX, vertY, vertZ, faces)) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return 1;
    }

    Vec3f position(0, 0, 3);
    float pitch = 0.0f;
    float yaw = M_PI / 4;  // 45 degrees rotation
    float roll = 0.0f;
    float fov = 60.0f;
    float aspect = (float)WIDTH / (float)HEIGHT;
    float near = 0.1f;
    float far = 100.0f;

    Vertices d_verts;
    int* d_faces;
    unsigned char* d_image;

    cudaMalloc(&d_verts.x, vertX.size() * sizeof(float));
    cudaMalloc(&d_verts.y, vertY.size() * sizeof(float));
    cudaMalloc(&d_verts.z, vertZ.size() * sizeof(float));
    cudaMalloc(&d_faces, faces.size() * sizeof(int));
    cudaMalloc(&d_image, WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    cudaMemcpy(d_verts.x, vertX.data(), vertX.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_verts.y, vertY.data(), vertY.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_verts.z, vertZ.data(), vertZ.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_image, 0, WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    int numFaces = faces.size() / 3;
    int numBlocks = (numFaces + BLOCK_SIZE - 1) / BLOCK_SIZE;

    drawWireframeKernel<<<numBlocks, BLOCK_SIZE>>>(d_verts, d_faces, numFaces, d_image,
                                                   position, pitch, yaw, roll,
                                                   fov, aspect, near, far);
    cudaDeviceSynchronize();

    unsigned char* h_image = new unsigned char[WIDTH * HEIGHT * 3];
    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", WIDTH, HEIGHT, 3, h_image, WIDTH * 3);

    delete[] h_image;
    cudaFree(d_verts.x);
    cudaFree(d_verts.y);
    cudaFree(d_verts.z);
    cudaFree(d_faces);
    cudaFree(d_image);

    return 0;
}