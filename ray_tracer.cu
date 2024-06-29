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

struct Object {
    Triangle* triangles;
    int num_triangles;
    unsigned char* texture;
    int tex_width;
    int tex_height;
    Mat4f model_matrix[2];  // Two model matrices, one for each scene
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

__global__ void ray_trace_kernel(Object* objects, int num_objects, unsigned char* output, int width, int height, int scene_index) {
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

    for (int obj = 0; obj < num_objects; obj++) {
        Object& object = objects[obj];
        for (int i = 0; i < object.num_triangles; i++) {
            Triangle transformed_triangle = object.triangles[i];
            for (int j = 0; j < 3; j++) {
                transformed_triangle.v[j] = object.model_matrix[scene_index].transform(object.triangles[i].v[j]);
                transformed_triangle.n[j] = object.model_matrix[scene_index].transformNormal(object.triangles[i].n[j]);
            }

            float t, u, v;
            if (ray_triangle_intersect(ray, transformed_triangle, t, u, v) && t < closest_t) {
                closest_t = t;

                // Barycentric coordinates
                float w = 1.0f - u - v;

                // Texture coordinates
                float tex_u = w * object.triangles[i].uv[0].u + u * object.triangles[i].uv[1].u + v * object.triangles[i].uv[2].u;
                float tex_v = w * object.triangles[i].uv[0].v + u * object.triangles[i].uv[1].v + v * object.triangles[i].uv[2].v;

                int tex_x = tex_u * object.tex_width;
                int tex_y = (1.0f - tex_v) * object.tex_height; // Flip V coordinate

                if (tex_x >= 0 && tex_x < object.tex_width && tex_y >= 0 && tex_y < object.tex_height) {
                    Vec3f tex_color;
                    tex_color.x = object.texture[(tex_y * object.tex_width + tex_x) * 3 + 0] / 255.0f;
                    tex_color.y = object.texture[(tex_y * object.tex_width + tex_x) * 3 + 1] / 255.0f;
                    tex_color.z = object.texture[(tex_y * object.tex_width + tex_x) * 3 + 2] / 255.0f;

                    // Interpolate vertex normals
                    Vec3f normal = (transformed_triangle.n[0] * w + transformed_triangle.n[1] * u + transformed_triangle.n[2] * v).normalize();

                    Vec3f light_dir = Vec3f(1, 1, 1).normalize();  // Light direction from top-right-front
                    float diffuse = max(0.0f, normal.dot(light_dir));

                    color = tex_color * (0.3f + 0.7f * diffuse);  // Adjusted ambient and diffuse factors
                }
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
        printf("Failed to open OBJ file: %s\n", filename);
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

    // Load African head
    std::vector<Triangle> african_head_triangles;
    load_obj("african_head.obj", african_head_triangles);
    printf("Loaded African head: %zu triangles\n", african_head_triangles.size());

    int african_head_tex_width, african_head_tex_height, african_head_tex_channels;
    unsigned char* african_head_texture = stbi_load("african_head_diffuse.tga", &african_head_tex_width, &african_head_tex_height, &african_head_tex_channels, 3);
    if (!african_head_texture) {
        printf("Failed to load African head texture\n");
        return 1;
    }
    printf("Loaded African head texture: %dx%d, %d channels\n", african_head_tex_width, african_head_tex_height, african_head_tex_channels);

    // Load drone
    std::vector<Triangle> drone_triangles;
    load_obj("drone.obj", drone_triangles);
    printf("Loaded drone: %zu triangles\n", drone_triangles.size());

    int drone_tex_width, drone_tex_height, drone_tex_channels;
    unsigned char* drone_texture = stbi_load("drone.png", &drone_tex_width, &drone_tex_height, &drone_tex_channels, 3);
    if (!drone_texture) {
        printf("Failed to load drone texture\n");
        return 1;
    }
    printf("Loaded drone texture: %dx%d, %d channels\n", drone_tex_width, drone_tex_height, drone_tex_channels);

    // Prepare objects
    Object objects[2];

    // African head
    objects[0].num_triangles = african_head_triangles.size();
    objects[0].tex_width = african_head_tex_width;
    objects[0].tex_height = african_head_tex_height;

    // African head model matrices
    // Scene 1
    objects[0].model_matrix[0].m[0][3] = -1.0f;  // Move left
    objects[0].model_matrix[0].m[1][3] = 0.0f;
    objects[0].model_matrix[0].m[2][3] = -3.0f;
    float angle = 3.14159f / 4.0f; // 45 degrees in radians
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    objects[0].model_matrix[0].m[0][0] = cos_angle;
    objects[0].model_matrix[0].m[0][2] = sin_angle;
    objects[0].model_matrix[0].m[2][0] = -sin_angle;
    objects[0].model_matrix[0].m[2][2] = cos_angle;

    // Scene 2
    objects[0].model_matrix[1].m[0][3] = 1.0f;  // Move right
    objects[0].model_matrix[1].m[1][3] = 0.5f;  // Move up
    objects[0].model_matrix[1].m[2][3] = -3.5f;
    angle = 3.14159f / 3.0f; // 60 degrees in radians
    cos_angle = cos(angle);
    sin_angle = sin(angle);
    objects[0].model_matrix[1].m[0][0] = cos_angle;
    objects[0].model_matrix[1].m[0][2] = sin_angle;
    objects[0].model_matrix[1].m[2][0] = -sin_angle;
    objects[0].model_matrix[1].m[2][2] = cos_angle;

    // Drone
    objects[1].num_triangles = drone_triangles.size();
    objects[1].tex_width = drone_tex_width;
    objects[1].tex_height = drone_tex_height;

    // Drone model matrices
    // Scene 1
    objects[1].model_matrix[0].m[0][3] = 1.0f;   // Move right
    objects[1].model_matrix[0].m[1][3] = 0.5f;   // Move up
    objects[1].model_matrix[0].m[2][3] = -2.5f;  // Move closer
    objects[1].model_matrix[0].m[0][0] = 0.1f;   // Scale down
    objects[1].model_matrix[0].m[1][1] = 0.1f;
    objects[1].model_matrix[0].m[2][2] = 0.1f;

    // Scene 2
    objects[1].model_matrix[1].m[0][3] = -1.5f;  // Move left
    objects[1].model_matrix[1].m[1][3] = -0.5f;  // Move down
    objects[1].model_matrix[1].m[2][3] = -2.0f;  // Move closer
    objects[1].model_matrix[1].m[0][0] = 0.15f;  // Scale up slightly
    objects[1].model_matrix[1].m[1][1] = 0.15f;
    objects[1].model_matrix[1].m[2][2] = 0.15f;

    // Allocate memory on device
    CHECK_CUDA(cudaMalloc(&objects[0].triangles, african_head_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&objects[0].texture, african_head_tex_width * african_head_tex_height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&objects[1].triangles, drone_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&objects[1].texture, drone_tex_width * drone_tex_height * 3 * sizeof(unsigned char)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(objects[0].triangles, african_head_triangles.data(), african_head_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(objects[0].texture, african_head_texture, african_head_tex_width * african_head_tex_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(objects[1].triangles, drone_triangles.data(), drone_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(objects[1].texture, drone_texture, drone_tex_width * drone_tex_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Allocate output buffers
    unsigned char* d_output1, *d_output2;
    CHECK_CUDA(cudaMalloc(&d_output1, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_output2, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemset(d_output1, 0, width * height * 3 * sizeof(unsigned char))); // Clear output buffer 1
    CHECK_CUDA(cudaMemset(d_output2, 0, width * height * 3 * sizeof(unsigned char))); // Clear output buffer 2

    // Copy objects to device
    Object* d_objects;
    CHECK_CUDA(cudaMalloc(&d_objects, 2 * sizeof(Object)));
    CHECK_CUDA(cudaMemcpy(d_objects, objects, 2 * sizeof(Object), cudaMemcpyHostToDevice));

    // Launch kernels
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    ray_trace_kernel<<<grid_size, block_size>>>(d_objects, 2, d_output1, width, height, 0);
    ray_trace_kernel<<<grid_size, block_size>>>(d_objects, 2, d_output2, width, height, 1);

    // Copy results back to host
    unsigned char* output1 = new unsigned char[width * height * 3];
    unsigned char* output2 = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output1, d_output1, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(output2, d_output2, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Save output images
    stbi_write_png("output_1.png", width, height, 3, output1, width * 3);
    stbi_write_png("output_2.png", width, height, 3, output2, width * 3);

    // Clean up
    delete[] output1;
    delete[] output2;
    stbi_image_free(african_head_texture);
    stbi_image_free(drone_texture);
    CHECK_CUDA(cudaFree(objects[0].triangles));
    CHECK_CUDA(cudaFree(objects[0].texture));
    CHECK_CUDA(cudaFree(objects[1].triangles));
    CHECK_CUDA(cudaFree(objects[1].texture));
    CHECK_CUDA(cudaFree(d_output1));
    CHECK_CUDA(cudaFree(d_output2));
    CHECK_CUDA(cudaFree(d_objects));

    return 0;
}