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
#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma nv_diag_default 550
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

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

struct Triangle {
    Vec3 v[3];
    Vec2 uv[3];
    Vec3 n[3];
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

__global__ void render_kernel(Triangle* transformed_triangles, int* triangle_offsets, int* triangle_counts,
                              unsigned char* textures, int* tex_widths, int* tex_heights,
                              unsigned char* output, float* zbuffer,
                              int width, int height, int num_objects, int num_scenes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int scene = blockIdx.z;
    if (x >= width || y >= height || scene >= num_scenes) return;

    int idx = (scene * height + y) * width + x;
    zbuffer[idx] = FLT_MAX;
    Vec3 color(0.2f, 0.2f, 0.2f);
    Vec3 light_dir(1, 1, 1);
    light_dir = light_dir.normalize();

    for (int obj = 0; obj < num_objects; obj++) {
        int triangle_offset = triangle_offsets[scene * num_objects + obj];
        for (int i = 0; i < triangle_counts[scene * num_objects + obj]; i++) {
            Triangle& tri = transformed_triangles[triangle_offset + i];
            
            Vec3 screen_coords[3];
            for (int j = 0; j < 3; j++) {
                screen_coords[j] = Vec3((tri.v[j].x + 1.0f) * width / 2.0f,
                                        (1.0f - tri.v[j].y) * height / 2.0f,
                                        tri.v[j].z);
            }

            Vec3 edge1 = screen_coords[1] - screen_coords[0];
            Vec3 edge2 = screen_coords[2] - screen_coords[0];
            Vec3 h = Vec3(x, y, 0) - screen_coords[0];
            float det = edge1.x * edge2.y - edge1.y * edge2.x;
            if (fabs(det) < 1e-6) continue;

            float u = (h.x * edge2.y - h.y * edge2.x) / det;
            float v = (edge1.x * h.y - edge1.y * h.x) / det;
            if (u < 0 || v < 0 || u + v > 1) continue;

            float z = screen_coords[0].z + u * (screen_coords[1].z - screen_coords[0].z) +
                      v * (screen_coords[2].z - screen_coords[0].z);
            if (z < zbuffer[idx]) {
                zbuffer[idx] = z;

                Vec2 uv = tri.uv[0] * (1-u-v) + tri.uv[1] * u + tri.uv[2] * v;
                int tex_x = uv.u * tex_widths[scene * num_objects + obj];
                int tex_y = (1.0f - uv.v) * tex_heights[scene * num_objects + obj];
                int tex_idx = (tex_y * tex_widths[scene * num_objects + obj] + tex_x) * 3;
                Vec3 tex_color(textures[tex_idx] / 255.0f,
                               textures[tex_idx + 1] / 255.0f,
                               textures[tex_idx + 2] / 255.0f);

                Vec3 normal = (tri.n[0] * (1-u-v) + tri.n[1] * u + tri.n[2] * v).normalize();
                float diffuse = max(0.0f, normal.dot(light_dir));
                
                color = tex_color * (0.3f + 0.7f * diffuse);
            }
        }
        textures += tex_widths[scene * num_objects + obj] * tex_heights[scene * num_objects + obj] * 3;
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

__global__ void transform_vertices_kernel(Triangle* input_triangles, Triangle* output_triangles, 
                                          int* triangle_offsets, int* triangle_counts, 
                                          Mat4* model_matrices, Mat4 projection, int num_objects, int num_scenes) {
    int scene = blockIdx.y;
    int obj = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scene >= num_scenes || obj >= num_objects) return;
    
    int offset = triangle_offsets[scene * num_objects + obj];
    int count = triangle_counts[scene * num_objects + obj];
    
    if (idx >= count) return;

    Triangle input_tri = input_triangles[offset + idx];
    Triangle& output_tri = output_triangles[offset + idx];
    Mat4 model = model_matrices[scene * num_objects + obj];
    Mat4 mp = projection * model;

    for (int j = 0; j < 3; j++) {
        output_tri.v[j] = mp.multiplyPoint(input_tri.v[j]);
        output_tri.n[j] = model.multiplyVector(input_tri.n[j]).normalize();
        output_tri.uv[j] = input_tri.uv[j];  // Copy UV coordinates
    }
}

int main() {
    const int width = 400, height = 300;
    const int num_objects = 2;
    const int num_scenes = 4;
    
    std::vector<std::vector<Triangle>> triangles(num_objects);
    std::vector<unsigned char*> textures(num_objects);
    std::vector<int> tex_widths(num_objects), tex_heights(num_objects);

    // Load objects and textures
    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    // Prepare projection matrix
    Mat4 projection = create_projection_matrix(3.14159f / 4.0f, (float)width / height, 0.1f, 100.0f);

    // Define model matrices for all scenes
    Mat4 model_matrices[num_scenes][num_objects];
    for (int scene = 0; scene < num_scenes; scene++) {
        for (int obj = 0; obj < num_objects; obj++) {
            model_matrices[scene][obj] = create_model_matrix_random();
        }
    }

    std::vector<Triangle> all_triangles;
    std::vector<int> triangle_offsets(num_scenes * num_objects), triangle_counts(num_scenes * num_objects);
    std::vector<unsigned char> all_textures;
    std::vector<int> all_tex_widths(num_scenes * num_objects), all_tex_heights(num_scenes * num_objects);

    for (int scene = 0; scene < num_scenes; scene++) {
        for (int i = 0; i < num_objects; i++) {
            triangle_offsets[scene * num_objects + i] = all_triangles.size();
            triangle_counts[scene * num_objects + i] = triangles[i].size();
            
            all_triangles.insert(all_triangles.end(), triangles[i].begin(), triangles[i].end());
            
            all_tex_widths[scene * num_objects + i] = tex_widths[i];
            all_tex_heights[scene * num_objects + i] = tex_heights[i];
            all_textures.insert(all_textures.end(), textures[i], textures[i] + tex_widths[i] * tex_heights[i] * 3);
        }
    }

    // Allocate GPU memory
    Triangle *d_input_triangles, *d_transformed_triangles;
    int *d_triangle_offsets, *d_triangle_counts, *d_tex_widths, *d_tex_heights;
    unsigned char *d_textures, *d_output;
    float* d_zbuffer;
    Mat4* d_model_matrices;

    CHECK_CUDA(cudaMalloc(&d_input_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_transformed_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_triangle_offsets, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_triangle_counts, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_textures, all_textures.size() * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, num_scenes * width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, num_scenes * width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_model_matrices, num_scenes * num_objects * sizeof(Mat4)));

    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_input_triangles, all_triangles.data(), all_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_offsets, triangle_offsets.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_counts, triangle_counts.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_textures, all_textures.data(), all_textures.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_widths, all_tex_widths.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_heights, all_tex_heights.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_model_matrices, model_matrices, num_scenes * num_objects * sizeof(Mat4), cudaMemcpyHostToDevice));

    // Transform vertices
    int max_triangles = *std::max_element(triangle_counts.begin(), triangle_counts.end());
    dim3 block_size(256);
    dim3 grid_size((max_triangles + block_size.x - 1) / block_size.x, num_scenes, num_objects);
    
    transform_vertices_kernel<<<grid_size, block_size>>>(
        d_input_triangles, d_transformed_triangles, d_triangle_offsets, d_triangle_counts,
        d_model_matrices, projection, num_objects, num_scenes);

    // Render scenes
    dim3 render_block_size(16, 16, 1);
    dim3 render_grid_size((width + render_block_size.x - 1) / render_block_size.x, 
                          (height + render_block_size.y - 1) / render_block_size.y, 
                          num_scenes);
    render_kernel<<<render_grid_size, render_block_size>>>(d_transformed_triangles, d_triangle_offsets, d_triangle_counts,
                                                           d_textures, d_tex_widths, d_tex_heights,
                                                           d_output, d_zbuffer, width, height, num_objects, num_scenes);

    // Copy result back to host and save
    std::vector<unsigned char> output(num_scenes * width * height * 3);
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, num_scenes * width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    for (int scene = 0; scene < num_scenes; scene++) {
        stbi_write_png(("output_" + std::to_string(scene) + ".png").c_str(), width, height, 3, 
                       output.data() + scene * width * height * 3, width * 3);
    }

    // Clean up GPU memory
    cudaFree(d_input_triangles);
    cudaFree(d_transformed_triangles);
    cudaFree(d_triangle_offsets);
    cudaFree(d_triangle_counts);
    cudaFree(d_textures);
    cudaFree(d_tex_widths);
    cudaFree(d_tex_heights);
    cudaFree(d_output);
    cudaFree(d_zbuffer);
    cudaFree(d_model_matrices);

    // Clean up
    for (auto texture : textures) {
        stbi_image_free(texture);
    }

    return 0;
}