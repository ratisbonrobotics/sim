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
    Vec3f n[3];
};

struct Mat4f {
    float m[4][4];

    __host__ __device__ Mat4f() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    __host__ __device__ Mat4f operator*(const Mat4f& other) const {
        Mat4f result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    __host__ __device__ Vec3f multiplyPoint(const Vec3f& v) const {
        float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3];
        float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3];
        float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3];
        float w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
        return Vec3f(x/w, y/w, z/w);
    }

    __host__ __device__ Vec3f multiplyVector(const Vec3f& v) const {
        float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
        float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
        float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;
        return Vec3f(x, y, z);
    }
};

__global__ void project_vertices_kernel(Triangle* triangles, int* triangle_counts, 
                                        Mat4f* model_matrices, Mat4f vp, int num_objects) {
    int obj_idx = blockIdx.x;
    int tri_idx = threadIdx.x + blockDim.x * blockIdx.y;

    if (obj_idx >= num_objects || tri_idx >= triangle_counts[obj_idx]) return;

    int triangle_offset = 0;
    for (int i = 0; i < obj_idx; i++) {
        triangle_offset += triangle_counts[i];
    }

    Triangle& tri = triangles[triangle_offset + tri_idx];
    Mat4f mvp = vp * model_matrices[obj_idx];

    for (int j = 0; j < 3; j++) {
        tri.v[j] = mvp.multiplyPoint(tri.v[j]);
        tri.n[j] = model_matrices[obj_idx].multiplyVector(tri.n[j]).normalize();
    }
}

__global__ void rasterize_kernel(Triangle* triangles, int* triangle_counts, 
                                 unsigned char* textures, int* tex_widths, int* tex_heights, 
                                 unsigned char* output, float* zbuffer, 
                                 int width, int height, int num_objects) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int flipped_y = height - 1 - y;
    int idx = flipped_y * width + x;
    zbuffer[idx] = FLT_MAX;

    Vec3f color(0.2f, 0.2f, 0.2f);  // Background color
    Vec3f light_dir = Vec3f(1, 1, 1).normalize();
    Vec3f P(x, flipped_y, 0);  // Current pixel position

    int triangle_offset = 0;
    int texture_offset = 0;

    for (int obj = 0; obj < num_objects; obj++) {
        for (int i = 0; i < triangle_counts[obj]; i++) {
            Triangle& tri = triangles[triangle_offset + i];
            
            // Transform vertices to screen space
            Vec3f screen_coords[3];
            for (int j = 0; j < 3; j++) {
                Vec3f v = tri.v[j];
                screen_coords[j] = Vec3f((v.x + 1.0f) * width / 2.0f, (1.0f - v.y) * height / 2.0f, v.z);
            }

            // Calculate barycentric coordinates
            Vec3f s[2];
            for (int k = 2; k--; ) {
                s[k].x = screen_coords[2][k] - screen_coords[0][k];
                s[k].y = screen_coords[1][k] - screen_coords[0][k];
                s[k].z = screen_coords[0][k] - P[k];
            }
            Vec3f u = s[0].cross(s[1]);
            Vec3f bc_screen;
            if (std::abs(u.z) > 1e-2)
                bc_screen = Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
            else
                bc_screen = Vec3f(-1, 1, 1);

            // Check if pixel is inside the triangle
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

            // Depth test
            float frag_depth = bc_screen.x * screen_coords[0].z + bc_screen.y * screen_coords[1].z + bc_screen.z * screen_coords[2].z;
            if (frag_depth < zbuffer[idx]) {
                zbuffer[idx] = frag_depth;

                // Texture sampling
                Vec2f uv = tri.uv[0] * bc_screen.x + tri.uv[1] * bc_screen.y + tri.uv[2] * bc_screen.z;
                int tex_x = uv.u * tex_widths[obj];
                int tex_y = (1.0f - uv.v) * tex_heights[obj];
                int tex_idx = texture_offset + (tex_y * tex_widths[obj] + tex_x) * 3;
                Vec3f tex_color(textures[tex_idx] / 255.0f, textures[tex_idx + 1] / 255.0f, textures[tex_idx + 2] / 255.0f);

                // Normal interpolation and lighting calculation
                Vec3f normal = (tri.n[0] * bc_screen.x + tri.n[1] * bc_screen.y + tri.n[2] * bc_screen.z).normalize();
                float diffuse = max(0.0f, normal.dot(light_dir));
                
                // Final color calculation
                color = tex_color * (0.3f + 0.7f * diffuse);
            }
        }
        triangle_offset += triangle_counts[obj];
        texture_offset += tex_widths[obj] * tex_heights[obj] * 3;
    }

    // Write final color to output buffer
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

Mat4f create_view_matrix(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
    Vec3f f = (center - eye).normalize();
    Vec3f s = f.cross(up).normalize();
    Vec3f u = s.cross(f);

    Mat4f result;
    result.m[0][0] = s.x;  result.m[0][1] = s.y;  result.m[0][2] = s.z;  result.m[0][3] = -s.dot(eye);
    result.m[1][0] = u.x;  result.m[1][1] = u.y;  result.m[1][2] = u.z;  result.m[1][3] = -u.dot(eye);
    result.m[2][0] = -f.x; result.m[2][1] = -f.y; result.m[2][2] = -f.z; result.m[2][3] = f.dot(eye);
    result.m[3][0] = 0.0f; result.m[3][1] = 0.0f; result.m[3][2] = 0.0f; result.m[3][3] = 1.0f;
    return result;
}

Mat4f create_model_matrix(float tx, float ty, float tz, float scale = 1.0f, float rotation = 0.0f) {
    Mat4f matrix;
    
    // Scale
    matrix.m[0][0] = matrix.m[1][1] = matrix.m[2][2] = scale;
    
    // Rotation (around Y-axis)
    float cos_r = cos(rotation);
    float sin_r = sin(rotation);
    matrix.m[0][0] = cos_r * scale;
    matrix.m[0][2] = -sin_r * scale;
    matrix.m[2][0] = sin_r * scale;
    matrix.m[2][2] = cos_r * scale;
    
    // Translation
    matrix.m[0][3] = tx;
    matrix.m[1][3] = ty;
    matrix.m[2][3] = tz;
    
    return matrix;
}

Mat4f create_perspective_matrix(float fov, float aspect, float near, float far) {
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

int main() {
    const int width = 340, height = 280;
    const int num_objects = 2;
    
    // Load objects and textures
    std::vector<Triangle> triangles[num_objects];
    unsigned char* textures[num_objects];
    int tex_widths[num_objects], tex_heights[num_objects];
    
    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    // Prepare model matrices, view and projection matrix
    Mat4f model_matrices[num_objects] = {
        create_model_matrix(-1.0f, 0.0f, -3.0f, 1.0f, 3.14159f * 1.75f), // African head
        create_model_matrix(1.0f, 0.5f, -2.5f, 0.1f)  // Drone
    };

    Mat4f* d_model_matrices;
    Mat4f vp = create_perspective_matrix(3.14159f / 4.0f, (float)width / height, 0.1f, 100.0f) * create_view_matrix(Vec3f(0, 0, 1), Vec3f(0, 0, 0), Vec3f(0, 1, 0));

    // Prepare GPU data
    Triangle* d_triangles;
    unsigned char* d_textures;
    int* d_triangle_counts, *d_tex_widths, *d_tex_heights;
    unsigned char* d_output;
    float* d_zbuffer;

    int total_triangles = triangles[0].size() + triangles[1].size();
    int total_texture_size = (tex_widths[0] * tex_heights[0] + tex_widths[1] * tex_heights[1]) * 3;
    int triangle_counts[num_objects] = {(int)triangles[0].size(), (int)triangles[1].size()};

    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(&d_model_matrices, num_objects * sizeof(Mat4f)));
    CHECK_CUDA(cudaMalloc(&d_triangles, total_triangles * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_textures, total_texture_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_triangle_counts, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, width * height * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_model_matrices, model_matrices, num_objects * sizeof(Mat4f), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangles, triangles[0].data(), triangles[0].size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangles + triangles[0].size(), triangles[1].data(), triangles[1].size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_textures, textures[0], tex_widths[0] * tex_heights[0] * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_textures + tex_widths[0] * tex_heights[0] * 3, textures[1], tex_widths[1] * tex_heights[1] * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_counts, triangle_counts, num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_widths, tex_widths, num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_heights, tex_heights, num_objects * sizeof(int), cudaMemcpyHostToDevice));

    // Project vertices and normals
    dim3 block_size(256);
    dim3 grid_size(num_objects, (std::max(triangle_counts[0], triangle_counts[1]) + block_size.x - 1) / block_size.x);
    project_vertices_kernel<<<grid_size, block_size>>>(d_triangles, d_triangle_counts, d_model_matrices, vp, num_objects);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Rasterize image
    block_size = dim3(16, 16);
    grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    rasterize_kernel<<<grid_size, block_size>>>(d_triangles, d_triangle_counts, d_textures, d_tex_widths, d_tex_heights, d_output, d_zbuffer, width, height, num_objects);

    // Copy result back to host and save
    unsigned char* output = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_png("output.png", width, height, 3, output, width * 3);

    // Clean up
    delete[] output;
    stbi_image_free(textures[0]);
    stbi_image_free(textures[1]);
    cudaFree(d_model_matrices);
    cudaFree(d_triangles);
    cudaFree(d_textures);
    cudaFree(d_triangle_counts);
    cudaFree(d_tex_widths);
    cudaFree(d_tex_heights);
    cudaFree(d_output);
    cudaFree(d_zbuffer);

    return 0;
}