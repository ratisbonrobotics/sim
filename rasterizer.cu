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

__global__ void render_kernel(Triangle* triangles, int* triangle_offsets, int* triangle_counts,
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
            Triangle& tri = triangles[triangle_offset + i];
            
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

Mat4 create_view_matrix(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = (center - eye).normalize();
    Vec3 s = f.cross(up).normalize();
    Vec3 u = s.cross(f);

    Mat4 result;
    result.m[0] = s.x;  result.m[1] = s.y;  result.m[2] = s.z;  result.m[3] = -s.dot(eye);
    result.m[4] = u.x;  result.m[5] = u.y;  result.m[6] = u.z;  result.m[7] = -u.dot(eye);
    result.m[8] = -f.x; result.m[9] = -f.y; result.m[10] = -f.z; result.m[11] = f.dot(eye);
    return result;
}

Mat4 create_projection_matrix(float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    Mat4 result;
    result.m[0] = 1.0f / (aspect * tanHalfFov);
    result.m[5] = 1.0f / tanHalfFov;
    result.m[10] = -(far + near) / (far - near);
    result.m[11] = -2.0f * far * near / (far - near);
    result.m[14] = -1.0f;
    result.m[15] = 0.0f;
    return result;
}

Mat4 create_model_matrix(float tx, float ty, float tz, float scale_x = 1.0f, float scale_y = 1.0f, float scale_z = 1.0f, float rotation = 0.0f) {
    Mat4 matrix;
    float cos_r = cos(rotation);
    float sin_r = sin(rotation);
    
    // Scale
    matrix.m[0] = cos_r * scale_x;
    matrix.m[5] = scale_y;
    matrix.m[10] = cos_r * scale_z;
    
    // Rotation (around Y-axis)
    matrix.m[2] = -sin_r * scale_z;
    matrix.m[8] = sin_r * scale_x;
    
    // Translation
    matrix.m[3] = tx;
    matrix.m[7] = ty;
    matrix.m[11] = tz;
    
    return matrix;
}

int main() {
    const int width = 800, height = 600;
    const int num_objects = 2;
    const int num_scenes = 2;
    
    std::vector<std::vector<Triangle>> triangles(num_objects);
    std::vector<unsigned char*> textures(num_objects);
    std::vector<int> tex_widths(num_objects), tex_heights(num_objects);

    // Load objects and textures
    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    // Prepare view and projection matrices
    Mat4 vp = create_projection_matrix(3.14159f / 4.0f, (float)width / height, 0.1f, 100.0f) * create_view_matrix(Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1, 0));

    // Define model matrices for both scenes
    Mat4 model_matrices[2][2] = {
        {create_model_matrix(-1.0f, 0.0f, -3.0f, 1.0f, 1.0f, 1.0f, 3.14159f * 1.75f),
         create_model_matrix(1.0f, 0.5f, -2.5f, 0.1f, 0.1f, 0.1f)},
        {create_model_matrix(-0.5f, 0.5f, -3.5f, 1.2f, 1.2f, 1.2f, 3.14159f * 1.5f),
         create_model_matrix(0.5f, -0.5f, -2.0f, 0.15f, 0.15f, 0.15f, 3.14159f * 0.5f)}
    };

    std::vector<Triangle> all_triangles;
    std::vector<int> triangle_offsets(num_scenes * num_objects), triangle_counts(num_scenes * num_objects);
    std::vector<unsigned char> all_textures;
    std::vector<int> all_tex_widths(num_scenes * num_objects), all_tex_heights(num_scenes * num_objects);

    for (int scene = 0; scene < num_scenes; scene++) {
        for (int i = 0; i < num_objects; i++) {
            Mat4 model_matrix = model_matrices[scene][i];
            triangle_offsets[scene * num_objects + i] = all_triangles.size();
            triangle_counts[scene * num_objects + i] = triangles[i].size();
            
            for (auto tri : triangles[i]) {
                for (int j = 0; j < 3; j++) {
                    tri.v[j] = (vp * model_matrix).multiplyPoint(tri.v[j]);
                    tri.n[j] = model_matrix.multiplyVector(tri.n[j]).normalize();
                }
                all_triangles.push_back(tri);
            }
            all_tex_widths[scene * num_objects + i] = tex_widths[i];
            all_tex_heights[scene * num_objects + i] = tex_heights[i];
            all_textures.insert(all_textures.end(), textures[i], textures[i] + tex_widths[i] * tex_heights[i] * 3);
        }
    }

    // Allocate GPU memory
    Triangle* d_triangles;
    int *d_triangle_offsets, *d_triangle_counts, *d_tex_widths, *d_tex_heights;
    unsigned char *d_textures, *d_output;
    float* d_zbuffer;

    CHECK_CUDA(cudaMalloc(&d_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_triangle_offsets, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_triangle_counts, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_textures, all_textures.size() * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, num_scenes * width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, num_scenes * width * height * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_triangles, all_triangles.data(), all_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_offsets, triangle_offsets.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_counts, triangle_counts.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_textures, all_textures.data(), all_textures.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_widths, all_tex_widths.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_heights, all_tex_heights.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));

    // Render image
    dim3 block_size(16, 16, 1);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y, 
                   num_scenes);
    render_kernel<<<grid_size, block_size>>>(d_triangles, d_triangle_offsets, d_triangle_counts,
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
    cudaFree(d_triangles);
    cudaFree(d_triangle_offsets);
    cudaFree(d_triangle_counts);
    cudaFree(d_textures);
    cudaFree(d_tex_widths);
    cudaFree(d_tex_heights);
    cudaFree(d_output);
    cudaFree(d_zbuffer);

    // Clean up
    for (auto texture : textures) {
        stbi_image_free(texture);
    }

    return 0;
}