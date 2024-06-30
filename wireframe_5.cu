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
};

__device__ __forceinline__ Vec3f rotate(const Vec3f& v, float cp, float sp, float cy, float sy, float cr, float sr) {
    Vec3f rotated;
    rotated.x = (cy * cr + sy * sp * sr) * v.x + (-cy * sr + sy * sp * cr) * v.y + (sy * cp) * v.z;
    rotated.y = (cp * sr) * v.x + (cp * cr) * v.y + (-sp) * v.z;
    rotated.z = (-sy * cr + cy * sp * sr) * v.x + (sy * sr + cy * sp * cr) * v.y + (cy * cp) * v.z;
    return rotated;
}

__device__ __forceinline__ Vec3f perspectiveProject(const Vec3f& v, float tanHalfFov, float aspect, float near, float far) {
    float x = v.x / (v.z * tanHalfFov);
    float y = v.y / (v.z * tanHalfFov * aspect);
    float z = (v.z - near) / (far - near);
    return Vec3f(x, y, z);
}

__device__ void drawLine(int x0, int y0, int x1, int y1, cudaSurfaceObject_t surfObj) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            surf2Dwrite(make_uchar4(255, 255, 255, 255), surfObj, x0 * sizeof(uchar4), y0);
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

__global__ void drawWireframeKernel(Vec3f* verts, int* faces, int numFaces, cudaSurfaceObject_t surfObj,
                                    Vec3f position, float cp, float sp, float cy, float sy, float cr, float sr,
                                    float tanHalfFov, float aspect, float near, float far) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFaces) return;

    Vec3f v[3];
    for (int j = 0; j < 3; j++) {
        int vertIdx = faces[idx * 3 + j];
        v[j] = rotate(verts[vertIdx], cp, sp, cy, sy, cr, sr);
        v[j].x -= position.x;
        v[j].y -= position.y;
        v[j].z -= position.z;
        v[j] = perspectiveProject(v[j], tanHalfFov, aspect, near, far);
    }

    for (int j = 0; j < 3; j++) {
        int x0 = (v[j].x + 1.0f) * width * 0.5f;
        int y0 = (v[j].y + 1.0f) * height * 0.5f;
        int x1 = (v[(j + 1) % 3].x + 1.0f) * width * 0.5f;
        int y1 = (v[(j + 1) % 3].y + 1.0f) * height * 0.5f;

        drawLine(x0, y0, x1, y1, surfObj);
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

    float tanHalfFov = tanf(fov * 0.5f * M_PI / 180.0f);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cy = cosf(yaw), sy = sinf(yaw);
    float cr = cosf(roll), sr = sinf(roll);

    Vec3f* d_verts;
    int* d_faces;

    cudaMalloc(&d_verts, vertices.size() * sizeof(Vec3f));
    cudaMalloc(&d_faces, faces.size() * sizeof(int));

    cudaMemcpy(d_verts, vertices.data(), vertices.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    cudaSurfaceObject_t surfObj;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaCreateSurfaceObject(&surfObj, &resDesc);

    int numFaces = faces.size() / 3;
    int threadsPerBlock = 256;
    int numBlocks = (numFaces + threadsPerBlock - 1) / threadsPerBlock;

    drawWireframeKernel<<<numBlocks, threadsPerBlock>>>(d_verts, d_faces, numFaces, surfObj,
                                                        position, cp, sp, cy, sy, cr, sr,
                                                        tanHalfFov, aspect, near, far);
    cudaDeviceSynchronize();

    unsigned char* h_image = new unsigned char[width * height * 4];
    cudaMemcpy2DFromArray(h_image, width * 4, cuArray, 0, 0, width * 4, height, cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", width, height, 4, h_image, width * 4);

    delete[] h_image;
    cudaFree(d_verts);
    cudaFree(d_faces);
    cudaFreeArray(cuArray);
    cudaDestroySurfaceObject(surfObj);

    return 0;
}