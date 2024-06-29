#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

struct Vec3f {
    float x, y, z;
    __host__ __device__ Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3f operator+(const Vec3f& v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3f operator-(const Vec3f& v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3f operator*(float f) const { return Vec3f(x * f, y * f, z * f); }
    __host__ __device__ float dot(const Vec3f& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3f cross(const Vec3f& v) const { return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    __host__ __device__ Vec3f normalize() const { float l = sqrt(x * x + y * y + z * z); return Vec3f(x / l, y / l, z / l); }
};

struct Triangle { Vec3f v[3]; Vec3f uv[3]; Vec3f n[3]; };
struct Ray { Vec3f origin, direction; };

struct Mat4f {
    float m[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    __host__ __device__ Vec3f transform(const Vec3f& v) const {
        float w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
        return Vec3f(
            (m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3]) / w,
            (m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3]) / w,
            (m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3]) / w
        );
    }
    __host__ __device__ Vec3f transformNormal(const Vec3f& n) const {
        return Vec3f(
            m[0][0] * n.x + m[0][1] * n.y + m[0][2] * n.z,
            m[1][0] * n.x + m[1][1] * n.y + m[1][2] * n.z,
            m[2][0] * n.x + m[2][1] * n.y + m[2][2] * n.z
        ).normalize();
    }
};

struct Object {
    Triangle* triangles;
    int num_triangles;
    unsigned char* texture;
    int tex_width, tex_height;
    Mat4f* model_matrices;
};

__device__ bool ray_triangle_intersect(const Ray& ray, const Triangle& triangle, float& t, float& u, float& v) {
    Vec3f edge1 = triangle.v[1] - triangle.v[0];
    Vec3f edge2 = triangle.v[2] - triangle.v[0];
    Vec3f h = ray.direction.cross(edge2);
    float a = edge1.dot(h);
    if (a > -1e-5f && a < 1e-5f) return false;
    float f = 1.0f / a;
    Vec3f s = ray.origin - triangle.v[0];
    u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;
    Vec3f q = s.cross(edge1);
    v = f * ray.direction.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * edge2.dot(q);
    return t > 1e-5f;
}

__global__ void ray_trace_kernel(Object* object, unsigned char* output, int width, int height, int num_scenes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int scene_index = blockIdx.z;
    if (x >= width || y >= height || scene_index >= num_scenes) return;

    float aspect_ratio = (float)width / height;
    float fov = 3.14159f / 4.0f;
    float tan_fov = tan(fov / 2.0f);

    Ray ray;
    ray.origin = Vec3f(0, 0, 3);
    ray.direction = Vec3f(
        ((2.0f * (x + 0.5f) / width - 1.0f) * aspect_ratio) * tan_fov,
        (1.0f - 2.0f * (y + 0.5f) / height) * tan_fov,
        -1
    ).normalize();

    Vec3f color(0.2f, 0.2f, 0.2f);
    float closest_t = 1e10f;

    for (int i = 0; i < object->num_triangles; i++) {
        Triangle transformed_triangle = object->triangles[i];
        for (int j = 0; j < 3; j++) {
            transformed_triangle.v[j] = object->model_matrices[scene_index].transform(object->triangles[i].v[j]);
            transformed_triangle.n[j] = object->model_matrices[scene_index].transformNormal(object->triangles[i].n[j]);
        }

        float t, u, v;
        if (ray_triangle_intersect(ray, transformed_triangle, t, u, v) && t < closest_t) {
            closest_t = t;
            float w = 1.0f - u - v;
            float tex_u = w * object->triangles[i].uv[0].x + u * object->triangles[i].uv[1].x + v * object->triangles[i].uv[2].x;
            float tex_v = w * object->triangles[i].uv[0].y + u * object->triangles[i].uv[1].y + v * object->triangles[i].uv[2].y;
            int tex_x = tex_u * object->tex_width;
            int tex_y = (1.0f - tex_v) * object->tex_height;

            if (tex_x >= 0 && tex_x < object->tex_width && tex_y >= 0 && tex_y < object->tex_height) {
                Vec3f tex_color(
                    object->texture[(tex_y * object->tex_width + tex_x) * 3 + 0] / 255.0f,
                    object->texture[(tex_y * object->tex_width + tex_x) * 3 + 1] / 255.0f,
                    object->texture[(tex_y * object->tex_width + tex_x) * 3 + 2] / 255.0f
                );

                Vec3f normal = (transformed_triangle.n[0] * w + transformed_triangle.n[1] * u + transformed_triangle.n[2] * v).normalize();
                Vec3f light_dir(1, 1, 1);
                light_dir = light_dir.normalize();
                float diffuse = max(0.0f, normal.dot(light_dir));
                color = tex_color * (0.3f + 0.7f * diffuse);
            }
        }
    }

    int output_index = (scene_index * height * width + y * width + x) * 3;
    output[output_index + 0] = static_cast<unsigned char>(min(color.x * 255.0f, 255.0f));
    output[output_index + 1] = static_cast<unsigned char>(min(color.y * 255.0f, 255.0f));
    output[output_index + 2] = static_cast<unsigned char>(min(color.z * 255.0f, 255.0f));
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Failed to open OBJ file: %s\n", filename);
        return;
    }

    std::vector<Vec3f> vertices, texcoords, normals;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            Vec3f v; iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (type == "vt") {
            Vec3f vt; iss >> vt.x >> vt.y;
            texcoords.push_back(vt);
        } else if (type == "vn") {
            Vec3f vn; iss >> vn.x >> vn.y >> vn.z;
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

Mat4f generate_random_model_matrix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::uniform_real_distribution<> angle_dis(0, 2 * M_PI);

    Mat4f matrix;
    matrix.m[0][3] = dis(gen) * 2;
    matrix.m[1][3] = dis(gen);
    matrix.m[2][3] = dis(gen) * 2 - 4;

    float angle = angle_dis(gen);
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    matrix.m[0][0] = cos_angle;
    matrix.m[0][2] = sin_angle;
    matrix.m[2][0] = -sin_angle;
    matrix.m[2][2] = cos_angle;

    return matrix;
}

int main() {
    const int width = 800, height = 600;
    const std::vector<int> scenes_to_test = {1, 2, 4, 8, 16, 32, 64};
    int max_num_scenes = *std::max_element(scenes_to_test.begin(), scenes_to_test.end());

    std::vector<Triangle> triangles;
    load_obj("african_head.obj", triangles);
    
    int tex_width, tex_height, tex_channels;
    unsigned char* texture = stbi_load("african_head_diffuse.tga", &tex_width, &tex_height, &tex_channels, 3);

    Object object;
    object.num_triangles = triangles.size();
    object.tex_width = tex_width;
    object.tex_height = tex_height;

    CHECK_CUDA(cudaMalloc(&object.triangles, triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&object.texture, tex_width * tex_height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemcpy(object.triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(object.texture, texture, tex_width * tex_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    std::vector<Mat4f> model_matrices(max_num_scenes);
    for (int i = 0; i < max_num_scenes; ++i) {
        model_matrices[i] = generate_random_model_matrix();
    }

    CHECK_CUDA(cudaMalloc(&object.model_matrices, max_num_scenes * sizeof(Mat4f)));
    CHECK_CUDA(cudaMemcpy(object.model_matrices, model_matrices.data(), max_num_scenes * sizeof(Mat4f), cudaMemcpyHostToDevice));

    unsigned char* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, max_num_scenes * width * height * 3 * sizeof(unsigned char)));

    Object* d_object;
    CHECK_CUDA(cudaMalloc(&d_object, sizeof(Object)));
    CHECK_CUDA(cudaMemcpy(d_object, &object, sizeof(Object), cudaMemcpyHostToDevice));

    for (int num_scenes : scenes_to_test) {
        auto start_time = std::chrono::high_resolution_clock::now();

        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                       (height + block_size.y - 1) / block_size.y, 
                       num_scenes);

        ray_trace_kernel<<<grid_size, block_size>>>(d_object, d_output, width, height, num_scenes);
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Rendered " << num_scenes << " scenes in " << duration.count() / 1000.0 << " seconds" << std::endl;

        unsigned char* output = new unsigned char[num_scenes * width * height * 3];
        CHECK_CUDA(cudaMemcpy(output, d_output, num_scenes * width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_scenes; ++i) {
            char filename[20];
            snprintf(filename, sizeof(filename), "output_%04d.png", i);
            stbi_write_png(filename, width, height, 3, output + i * width * height * 3, width * 3);
        }

        delete[] output;
    }

    // Clean up
    CHECK_CUDA(cudaFree(object.triangles));
    CHECK_CUDA(cudaFree(object.texture));
    CHECK_CUDA(cudaFree(object.model_matrices));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_object));

    stbi_image_free(texture);

    return 0;
}