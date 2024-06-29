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
};

struct Vec2f {
    float u, v;
};

struct Triangle {
    Vec3f v[3];
    Vec2f uv[3];
    Vec3f n[3];  // Vertex normals
};

struct Ray {
    Vec3f origin;
    Vec3f direction;
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

__device__ bool ray_triangle_intersect(const Ray& ray, const Triangle& triangle, float& t, float& u, float& v) {
    Vec3f edge1 = triangle.v[1] - triangle.v[0];
    Vec3f edge2 = triangle.v[2] - triangle.v[0];
    Vec3f h = ray.direction.cross(edge2);
    float a = edge1.dot(h);

    if (a > -1e-5 && a < 1e-5) return false;

    float f = 1.0f / a;
    Vec3f s = ray.origin - triangle.v[0];
    u = f * s.dot(h);

    if (u < 0.0f || u > 1.0f) return false;

    Vec3f q = s.cross(edge1);
    v = f * ray.direction.dot(q);

    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * edge2.dot(q);

    return t > 1e-5;
}

__global__ void ray_trace_kernel(Triangle* triangles, int num_triangles, unsigned char* texture, int tex_width, int tex_height, unsigned char* output, int width, int height, Mat4f model_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float aspect_ratio = (float)width / height;
    float fov = 3.14159f / 4.0f;
    float tan_fov = tan(fov / 2.0f);

    float camera_x = ((2.0f * (x + 0.5f) / width - 1.0f) * aspect_ratio) * tan_fov;
    float camera_y = (1.0f - 2.0f * (y + 0.5f) / height) * tan_fov;

    Ray ray;
    ray.origin = Vec3f(0, 0, 3);  // Move camera to positive z-axis
    ray.direction = Vec3f(camera_x, camera_y, -1).normalize();  // Look towards negative z-axis

    Vec3f color(0.2f, 0.2f, 0.2f); // Ambient light
    float closest_t = FLT_MAX;

    for (int i = 0; i < num_triangles; i++) {
        Triangle transformed_triangle = triangles[i];
        for (int j = 0; j < 3; j++) {
            transformed_triangle.v[j] = model_matrix.transform(triangles[i].v[j]);
            transformed_triangle.n[j] = model_matrix.transformNormal(triangles[i].n[j]);
        }

        float t, u, v;
        if (ray_triangle_intersect(ray, transformed_triangle, t, u, v) && t < closest_t) {
            closest_t = t;

            // Barycentric coordinates
            float w = 1.0f - u - v;

            // Texture coordinates
            float tex_u = w * triangles[i].uv[0].u + u * triangles[i].uv[1].u + v * triangles[i].uv[2].u;
            float tex_v = w * triangles[i].uv[0].v + u * triangles[i].uv[1].v + v * triangles[i].uv[2].v;

            int tex_x = tex_u * tex_width;
            int tex_y = (1.0f - tex_v) * tex_height; // Flip V coordinate

            if (tex_x >= 0 && tex_x < tex_width && tex_y >= 0 && tex_y < tex_height) {
                Vec3f tex_color;
                tex_color.x = texture[(tex_y * tex_width + tex_x) * 3 + 0] / 255.0f;
                tex_color.y = texture[(tex_y * tex_width + tex_x) * 3 + 1] / 255.0f;
                tex_color.z = texture[(tex_y * tex_width + tex_x) * 3 + 2] / 255.0f;

                // Interpolate vertex normals
                Vec3f normal = (transformed_triangle.n[0] * w + transformed_triangle.n[1] * u + transformed_triangle.n[2] * v).normalize();

                Vec3f light_dir = Vec3f(1, 1, 1).normalize();  // Light direction from top-right-front
                float diffuse = max(0.0f, normal.dot(light_dir));

                color = tex_color * (0.3f + 0.7f * diffuse);  // Adjusted ambient and diffuse factors
            }
        }
    }

    output[(y * width + x) * 3 + 0] = static_cast<unsigned char>(min(color.x * 255.0f, 255.0f));
    output[(y * width + x) * 3 + 1] = static_cast<unsigned char>(min(color.y * 255.0f, 255.0f));
    output[(y * width + x) * 3 + 2] = static_cast<unsigned char>(min(color.z * 255.0f, 255.0f));
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Failed to open OBJ file\n");
        return;
    }

    std::vector<Vec3f> vertices;
    std::vector<Vec2f> texcoords;
    std::vector<Vec3f> normals;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
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
    const int width = 800;
    const int height = 600;

    std::vector<Triangle> triangles;
    load_obj("african_head.obj", triangles);

    printf("Loaded %zu triangles\n", triangles.size());

    int tex_width, tex_height, tex_channels;
    unsigned char* texture = stbi_load("african_head_diffuse.tga", &tex_width, &tex_height, &tex_channels, 3);
    if (!texture) {
        printf("Failed to load texture\n");
        return 1;
    }
    printf("Loaded texture: %dx%d, %d channels\n", tex_width, tex_height, tex_channels);

    Triangle* d_triangles;
    unsigned char* d_texture;
    unsigned char* d_output;

    CHECK_CUDA(cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_texture, tex_width * tex_height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));

    CHECK_CUDA(cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_texture, texture, tex_width * tex_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, width * height * 3 * sizeof(unsigned char))); // Clear output buffer

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Create model matrix for translation and rotation
    Mat4f model_matrix;
    // Translate the model
    model_matrix.m[0][3] = 0.0f;  // Move 0 units along x-axis
    model_matrix.m[1][3] = 0.0f; // Move 0 units along y-axis
    model_matrix.m[2][3] = -3.0f;  // No movement along z-axis

    // Rotate the model (example: rotate 45 degrees around y-axis)
    float angle = 3.14159f / 4.0f; // 45 degrees in radians
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    model_matrix.m[0][0] = cos_angle;
    model_matrix.m[0][2] = sin_angle;
    model_matrix.m[2][0] = -sin_angle;
    model_matrix.m[2][2] = cos_angle;

    ray_trace_kernel<<<grid_size, block_size>>>(d_triangles, triangles.size(), d_texture, tex_width, tex_height, d_output, width, height, model_matrix);

    unsigned char* output = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    stbi_write_png("output.png", width, height, 3, output, width * 3);

    delete[] output;
    stbi_image_free(texture);
    CHECK_CUDA(cudaFree(d_triangles));
    CHECK_CUDA(cudaFree(d_texture));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}