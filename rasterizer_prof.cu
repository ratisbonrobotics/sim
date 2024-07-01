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

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

struct Vec3f {
    float x, y, z;
    __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3f operator+(const Vec3f& v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3f operator-(const Vec3f& v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3f operator*(float f) const { return Vec3f(x * f, y * f, z * f); }
    __host__ __device__ float dot(const Vec3f& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3f cross(const Vec3f& v) const { return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    __host__ __device__ Vec3f normalize() const { float l = sqrt(x * x + y * y + z * z); return Vec3f(x / l, y / l, z / l); }
    __host__ __device__ float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    __host__ __device__ const float& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};


struct Vec2f {
    float u, v;
    __host__ __device__ Vec2f() : u(0), v(0) {}
    __host__ __device__ Vec2f(float u, float v) : u(u), v(v) {}
    __host__ __device__ Vec2f operator*(float f) const { return Vec2f(u * f, v * f); }
    __host__ __device__ Vec2f operator+(const Vec2f& other) const { return Vec2f(u + other.u, v + other.v); }
};

struct Triangle {
    Vec3f v[3];
    Vec2f uv[3];
    Vec3f n[3];  // Vertex normals
};

struct Mat4f {
    float m[4][4];

    __host__ __device__ Mat4f() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    __host__ __device__ Vec3f transform(const Vec3f& v) const {
        float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3];
        float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3];
        float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3];
        float w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
        return Vec3f(x/w, y/w, z/w);
    }

    __host__ __device__ Vec3f transformNormal(const Vec3f& n) const {
        float x = m[0][0] * n.x + m[0][1] * n.y + m[0][2] * n.z;
        float y = m[1][0] * n.x + m[1][1] * n.y + m[1][2] * n.z;
        float z = m[2][0] * n.x + m[2][1] * n.y + m[2][2] * n.z;
        return Vec3f(x, y, z).normalize();
    }
};

__device__ Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f s[2];
    for (int i = 2; i--; ) {
        s[i].x = C[i] - A[i];
        s[i].y = B[i] - A[i];
        s[i].z = A[i] - P[i];
    }
    Vec3f u = s[0].cross(s[1]);
    if (std::abs(u.z) > 1e-2)
        return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
    return Vec3f(-1, 1, 1);
}

__host__ __device__ Mat4f perspective(float fov, float aspect, float near, float far) {
    Mat4f result;
    float tanHalfFov = tan(fov / 2.0f);
    
    result.m[0][0] = 1.0f / (aspect * tanHalfFov);
    result.m[1][1] = 1.0f / tanHalfFov;
    result.m[2][2] = -(far + near) / (far - near);
    result.m[2][3] = -2.0f * far * near / (far - near);
    result.m[3][2] = -1.0f;
    result.m[3][3] = 0.0f;
    
    return result;
}

__global__ void rasterize_kernel(Vec3f* projected_vertices, Triangle* triangles, int* triangle_counts, 
                                 unsigned char* textures, int* tex_widths, int* tex_heights, 
                                 Mat4f* model_matrices, unsigned char* output, float* zbuffer, 
                                 int width, int height, int num_objects) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int flipped_y = height - 1 - y;
    int idx = flipped_y * width + x;
    zbuffer[idx] = FLT_MAX;

    Vec3f color(0.2f, 0.2f, 0.2f); // Ambient light
    Vec3f light_dir = Vec3f(1, 1, 1).normalize();
    Vec3f P(x, flipped_y, 0);

    int vertex_offset = 0;
    int triangle_offset = 0;
    int texture_offset = 0;

    for (int obj = 0; obj < num_objects; obj++) {
        for (int i = 0; i < triangle_counts[obj]; i++) {
            Triangle& tri = triangles[triangle_offset + i];
            
            Vec3f screen_coords[3];
            for (int j = 0; j < 3; j++) {
                Vec3f v = projected_vertices[vertex_offset + i*3 + j];
                screen_coords[j] = Vec3f((v.x + 1.0f) * width / 2.0f, (1.0f - v.y) * height / 2.0f, v.z);
            }

            Vec3f bc_screen = barycentric(screen_coords[0], screen_coords[1], screen_coords[2], P);
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

            float frag_depth = bc_screen.x * screen_coords[0].z + bc_screen.y * screen_coords[1].z + bc_screen.z * screen_coords[2].z;
            if (frag_depth < zbuffer[idx]) {
                zbuffer[idx] = frag_depth;

                Vec2f uv = tri.uv[0] * bc_screen.x + tri.uv[1] * bc_screen.y + tri.uv[2] * bc_screen.z;
                int tex_x = uv.u * tex_widths[obj];
                int tex_y = (1.0f - uv.v) * tex_heights[obj];

                int tex_idx = texture_offset + (tex_y * tex_widths[obj] + tex_x) * 3;
                Vec3f tex_color(textures[tex_idx] / 255.0f, textures[tex_idx + 1] / 255.0f, textures[tex_idx + 2] / 255.0f);

                Vec3f normal = (tri.n[0] * bc_screen.x + tri.n[1] * bc_screen.y + tri.n[2] * bc_screen.z).normalize();
                normal = model_matrices[obj].transformNormal(normal);

                float diffuse = max(0.0f, normal.dot(light_dir));
                color = tex_color * (0.3f + 0.7f * diffuse);
            }
        }
        vertex_offset += triangle_counts[obj] * 3;
        triangle_offset += triangle_counts[obj];
        texture_offset += tex_widths[obj] * tex_heights[obj] * 3;
    }

    output[idx * 3 + 0] = static_cast<unsigned char>(min(color.x * 255.0f, 255.0f));
    output[idx * 3 + 1] = static_cast<unsigned char>(min(color.y * 255.0f, 255.0f));
    output[idx * 3 + 2] = static_cast<unsigned char>(min(color.z * 255.0f, 255.0f));
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Failed to open OBJ file: %s\n", filename);
        return;
    }

    std::vector<Vec3f> vertices, normals;
    std::vector<Vec2f> texcoords;
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

int main() {
    const int width = 640, height = 480;
    const int num_objects = 2;
    
    // Load objects and textures
    std::vector<Triangle> triangles[num_objects];
    unsigned char* textures[num_objects];
    int tex_widths[num_objects], tex_heights[num_objects];
    
    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    for (int i = 0; i < num_objects; i++) {
        if (!textures[i]) {
            printf("Failed to load texture for object %d\n", i);
            return 1;
        }
    }

    // Prepare model matrices
    Mat4f model_matrices[num_objects] = {
        Mat4f(), // African head
        Mat4f()  // Drone
    };
    
    // African head transformation
    model_matrices[0].m[0][3] = -1.0f;
    model_matrices[0].m[2][3] = -3.0f;
    float angle = 3.14159f / 4.0f;
    model_matrices[0].m[0][0] = model_matrices[0].m[2][2] = cos(angle);
    model_matrices[0].m[0][2] = -(model_matrices[0].m[2][0] = -sin(angle));

    // Drone transformation
    model_matrices[1].m[0][3] = 1.0f;
    model_matrices[1].m[1][3] = 0.5f;
    model_matrices[1].m[2][3] = -2.5f;
    model_matrices[1].m[0][0] = model_matrices[1].m[1][1] = model_matrices[1].m[2][2] = 0.1f;

    // Prepare projection matrix
    Mat4f proj = perspective(3.14159f / 4.0f, (float)width / height, 0.1f, 100.0f);

    // Project vertices
    std::vector<Vec3f> projected_vertices;
    for (int i = 0; i < num_objects; i++) {
        for (const auto& tri : triangles[i]) {
            for (int j = 0; j < 3; j++) {
                projected_vertices.push_back(proj.transform(model_matrices[i].transform(tri.v[j])));
            }
        }
    }

    // Prepare GPU data
    Triangle* d_triangles;
    unsigned char* d_textures;
    int* d_triangle_counts, *d_tex_widths, *d_tex_heights;
    Mat4f* d_model_matrices;
    Vec3f* d_projected_vertices;
    unsigned char* d_output;
    float* d_zbuffer;

    int total_triangles = triangles[0].size() + triangles[1].size();
    int total_texture_size = (tex_widths[0] * tex_heights[0] + tex_widths[1] * tex_heights[1]) * 3;

    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(&d_triangles, total_triangles * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_textures, total_texture_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_triangle_counts, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_model_matrices, num_objects * sizeof(Mat4f)));
    CHECK_CUDA(cudaMalloc(&d_projected_vertices, projected_vertices.size() * sizeof(Vec3f)));
    CHECK_CUDA(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, width * height * sizeof(float)));

    // Copy data to GPU
    int triangle_offset = 0, texture_offset = 0;
    for (int i = 0; i < num_objects; i++) {
        CHECK_CUDA(cudaMemcpy(d_triangles + triangle_offset, triangles[i].data(), triangles[i].size() * sizeof(Triangle), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_textures + texture_offset, textures[i], tex_widths[i] * tex_heights[i] * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
        triangle_offset += triangles[i].size();
        texture_offset += tex_widths[i] * tex_heights[i] * 3;
    }

    int triangle_counts[num_objects] = {(int)triangles[0].size(), (int)triangles[1].size()};
    CHECK_CUDA(cudaMemcpy(d_triangle_counts, triangle_counts, num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_widths, tex_widths, num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_heights, tex_heights, num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_model_matrices, model_matrices, num_objects * sizeof(Mat4f), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_projected_vertices, projected_vertices.data(), projected_vertices.size() * sizeof(Vec3f), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    const int num_runs = 100; // Number of times to run the kernel
    float total_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start);

        rasterize_kernel<<<grid_size, block_size>>>(d_projected_vertices, d_triangles, d_triangle_counts, 
                                                    d_textures, d_tex_widths, d_tex_heights, 
                                                    d_model_matrices, d_output, d_zbuffer, width, height, num_objects);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        printf("Run %d: %.2f ms\n", run + 1, milliseconds);
    }

    float average_time = total_time / num_runs;
    printf("Average kernel execution time: %.2f ms\n", average_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host and save
    unsigned char* output = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_png("output.png", width, height, 3, output, width * 3);

    // Clean up
    delete[] output;
    for (int i = 0; i < num_objects; i++) stbi_image_free(textures[i]);
    cudaFree(d_triangles);
    cudaFree(d_textures);
    cudaFree(d_triangle_counts);
    cudaFree(d_tex_widths);
    cudaFree(d_tex_heights);
    cudaFree(d_model_matrices);
    cudaFree(d_projected_vertices);
    cudaFree(d_output);
    cudaFree(d_zbuffer);

    return 0;
}