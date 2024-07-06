#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <algorithm>
#pragma nv_diag_suppress 611
#include <opencv2/opencv.hpp>
#pragma nv_diag_default 611
#include <cfloat>
#include <cmath>

#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma nv_diag_default 550

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    __host__ __device__ Vec3 normalize() const { float l = sqrt(x * x + y * y + z * z); return Vec3(x / l, y / l, z / l); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
};

struct Vec2 {
    float u, v;
    __host__ __device__ Vec2(float u = 0, float v = 0) : u(u), v(v) {}
    __host__ __device__ Vec2 operator*(float f) const { return Vec2(u * f, v * f); }
    __host__ __device__ Vec2 operator+(const Vec2& other) const { return Vec2(u + other.u, v + other.v); }
};

struct Mat3 {
    float m[9];
    __host__ __device__ Mat3() { for (int i = 0; i < 9; i++) m[i] = (i % 4 == 0) ? 1.0f : 0.0f; }
    __host__ __device__ Mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22) {
        m[0] = m00; m[1] = m01; m[2] = m02; m[3] = m10; m[4] = m11; m[5] = m12; m[6] = m20; m[7] = m21; m[8] = m22;
    }
    __host__ __device__ static Mat3 identity() { return Mat3(); }
    __host__ __device__ static Mat3 diag(float a, float b, float c) { Mat3 result; result.m[0] = a; result.m[4] = b; result.m[8] = c; return result; }
    __host__ __device__ Mat3 operator+(const Mat3& other) const { Mat3 result; for (int i = 0; i < 9; i++) result.m[i] = m[i] + other.m[i]; return result; }
    __host__ __device__ Mat3 operator*(const Mat3& other) const {
        Mat3 result;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            result.m[i*3 + j] = 0;
            for (int k = 0; k < 3; k++) result.m[i*3 + j] += m[i*3 + k] * other.m[k*3 + j];
        }
        return result;
    }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(m[0]*v.x + m[1]*v.y + m[2]*v.z, m[3]*v.x + m[4]*v.y + m[5]*v.z, m[6]*v.x + m[7]*v.y + m[8]*v.z); }
    __host__ __device__ Mat3& operator+=(const Mat3& other) { for (int i = 0; i < 9; i++) m[i] += other.m[i]; return *this; }
    __host__ __device__ Mat3 operator*(float scalar) const {
        Mat3 result;
        for (int i = 0; i < 9; i++) {
            result.m[i] = m[i] * scalar;
        }
        return result;
    }
    __host__ __device__ Vec3 multiplyVector(const Vec3& v) const {
        return Vec3(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z
        );
    }
};

struct Mat4 {
    float m[16];
    __host__ __device__ Mat4() { for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f; }
    __host__ __device__ Mat4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13,
                             float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33) {
        m[0] = m00; m[1] = m01; m[2] = m02; m[3] = m03; m[4] = m10; m[5] = m11; m[6] = m12; m[7] = m13;
        m[8] = m20; m[9] = m21; m[10] = m22; m[11] = m23; m[12] = m30; m[13] = m31; m[14] = m32; m[15] = m33;
    }
    __host__ __device__ Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
            result.m[i*4 + j] = 0;
            for (int k = 0; k < 4; k++) result.m[i*4 + j] += m[i*4 + k] * other.m[k*4 + j];
        }
        return result;
    }
    __host__ __device__ Vec3 multiplyPoint(const Vec3& v) const {
        float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];
        return Vec3((m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3]) / w,
                    (m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7]) / w,
                    (m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]) / w);
    }
    __host__ __device__ Vec3 multiplyVector(const Vec3& v) const { return Vec3(m[0] * v.x + m[1] * v.y + m[2] * v.z, m[4] * v.x + m[5] * v.y + m[6] * v.z, m[8] * v.x + m[9] * v.y + m[10] * v.z); }
    __host__ __device__ static Mat4 identity() { return Mat4(); }
    __host__ __device__ static Mat4 rotationY(float angle) {
        float c = cos(angle), s = sin(angle);
        return Mat4(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1);
    }
    __host__ __device__ void setTranslation(const Vec3& t) { m[3] = t.x; m[7] = t.y; m[11] = t.z; }
    __host__ __device__ void setRotation(const Mat3& r) { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m[i*4 + j] = r.m[i*3 + j]; }
    __host__ __device__ static Mat4 scale(float x, float y, float z) { return Mat4(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1); }
};

struct Triangle { Vec3 v[3]; Vec2 uv[3]; Vec3 n[3]; };

Mat4 create_projection_matrix(float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    float f = 1.0f / tanHalfFov;
    float nf = 1.0f / (near - far);
    return Mat4(f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, 2.0f * far * near * nf, 0, 0, -1, 0);
}

Mat4 create_model_matrix_random() {
    static std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> dis_pos(-1.0f, 1.0f), dis_scale(0.8f, 1.3f), dis_rot(0.0f, 2.0f * 3.14159f);
    float tx = dis_pos(gen), ty = dis_pos(gen), tz = dis_pos(gen) - 5.0f, scale = dis_scale(gen), rotation = dis_rot(gen);
    float cos_r = cos(rotation), sin_r = sin(rotation);
    return Mat4(cos_r * scale, 0, -sin_r * scale, tx, 0, scale, 0, ty, sin_r * scale, 0, cos_r * scale, tz, 0, 0, 0, 1);
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) { printf("Failed to open OBJ file: %s\n", filename); return; }
    std::vector<Vec3> vertices, normals;
    std::vector<Vec2> texcoords;
    std::string line, type;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> type;
        if (type == "v") { Vec3 v; iss >> v.x >> v.y >> v.z; vertices.push_back(v); }
        else if (type == "vt") { Vec2 vt; iss >> vt.u >> vt.v; texcoords.push_back(vt); }
        else if (type == "vn") { Vec3 vn; iss >> vn.x >> vn.y >> vn.z; normals.push_back(vn); }
        else if (type == "f") {
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

__host__ __device__ Mat3 skew(const Vec3& v) { return Mat3(0, -v.z, v.y, v.z, 0, -v.x, -v.y, v.x, 0); }

__global__ void render_kernel(Triangle* triangles, int* offsets, int* counts,
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
    Vec3 light_dir = Vec3(1, 1, 1).normalize();

    for (int obj = 0; obj < num_objects; obj++) {
        int offset = offsets[scene * num_objects + obj];
        int count = counts[scene * num_objects + obj];
        for (int i = 0; i < count; i++) {
            Triangle& tri = triangles[offset + i];
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

__global__ void transform_vertices_kernel(Triangle* in, Triangle* out, 
                                          int* offsets, int* counts, 
                                          Mat4* models, Mat4 projection, int num_objects, int num_scenes) {
    int scene = blockIdx.y, obj = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scene >= num_scenes || obj >= num_objects) return;
    
    int offset = offsets[scene * num_objects + obj];
    int count = counts[scene * num_objects + obj];
    if (idx >= count) return;

    Triangle in_tri = in[offset + idx];
    Triangle& out_tri = out[offset + idx];
    Mat4 model = models[scene * num_objects + obj];
    Mat4 mp = projection * model;

    for (int j = 0; j < 3; j++) {
        out_tri.v[j] = mp.multiplyPoint(in_tri.v[j]);
        out_tri.n[j] = model.multiplyVector(in_tri.n[j]).normalize();
        out_tri.uv[j] = in_tri.uv[j];
    }
}

void drone_controller(const Vec3& ang_vel_B, const Vec3& lin_vel_W, const Vec3& lin_pos_W, const Mat3& R_W_B,
                             const Vec3& target_pos, std::vector<float>& omega) {
    const float k_p_xy = 0.5f, k_d_xy = 0.2f;
    const float k_p_z = 2.0f, k_d_z = 0.5f;
    const float k_p_ang_vel = 0.1f;
    const float omega_min = 30.0f, omega_max = 70.0f;

    Vec3 error_pos = target_pos - lin_pos_W;
    Vec3 error_vel = -lin_vel_W;

    Vec3 f_W_desired = Vec3(
        k_p_xy * error_pos.x + k_d_xy * error_vel.x,
        k_p_xy * error_pos.y + k_d_xy * error_vel.y,
        k_p_z * error_pos.z + k_d_z * error_vel.z
    );

    Vec3 z_B = R_W_B.multiplyVector(Vec3(0, 0, 1));
    float f_B_thrust_desired = f_W_desired.dot(z_B);

    Vec3 x_C_desired = Vec3(f_W_desired.x, f_W_desired.y, 0).normalize();
    Vec3 y_C_desired = z_B.cross(x_C_desired).normalize();
    Vec3 x_B = R_W_B.multiplyVector(Vec3(1, 0, 0));
    Vec3 y_B = R_W_B.multiplyVector(Vec3(0, 1, 0));

    Vec3 error_ang_vel(
        k_p_ang_vel * y_B.dot(x_C_desired),
        k_p_ang_vel * x_B.dot(y_C_desired),
        0
    );

    Vec3 tau_B_desired = -error_ang_vel;

    float omega_sq[4];
    omega_sq[0] = (f_B_thrust_desired + tau_B_desired.y) / 4.0f;
    omega_sq[1] = (f_B_thrust_desired - tau_B_desired.x) / 4.0f;
    omega_sq[2] = (f_B_thrust_desired - tau_B_desired.y) / 4.0f;
    omega_sq[3] = (f_B_thrust_desired + tau_B_desired.x) / 4.0f;

    for (int i = 0; i < 4; i++) {
        omega[i] = std::sqrt(std::max(omega_sq[i], 0.0f));
        omega[i] = std::max(std::min(omega[i], omega_max), omega_min);
    }
}

void update_drone_dynamics(std::vector<Vec3>& ang_vel_B, std::vector<Vec3>& lin_vel_W,
                           std::vector<Vec3>& lin_pos_W, std::vector<Mat3>& R_W_B,
                           std::vector<float>& omega, Mat4 model_matrices[][2], float dt) {
    const float k_f = 0.0004905f, k_m = 0.00004905f, L = 0.25f;
    const float I[3] = {0.01f, 0.02f, 0.01f}, g = 9.81f, m = 0.5f;
    const float omega_min = 30.0f, omega_max = 70.0f;

    for (int scene = 0; scene < ang_vel_B.size(); scene++) {
        for (int i = 0; i < 4; i++) omega[i] = std::max(std::min(omega[i], omega_max), omega_min);

        float F[4], M[4];
        for (int i = 0; i < 4; i++) {
            F[i] = k_f * omega[i] * std::abs(omega[i]);
            M[i] = k_m * omega[i] * std::abs(omega[i]);
        }

        Vec3 f_B_thrust(0, F[0] + F[1] + F[2] + F[3], 0);
        Vec3 tau_B_drag(0, M[0] - M[1] + M[2] - M[3], 0);
        Vec3 tau_B_thrust = 
            Vec3(-L, 0, L).cross(Vec3(0, F[0], 0)) +
            Vec3(L, 0, L).cross(Vec3(0, F[1], 0)) +
            Vec3(L, 0, -L).cross(Vec3(0, F[2], 0)) +
            Vec3(-L, 0, -L).cross(Vec3(0, F[3], 0));
        Vec3 tau_B = tau_B_drag + tau_B_thrust;

        Vec3 lin_acc_W = Vec3(0, -g * m, 0) + R_W_B[scene] * f_B_thrust;
        lin_acc_W = lin_acc_W * (1.0f / m);

        Mat3 I_mat = Mat3::diag(I[0], I[1], I[2]);
        Vec3 ang_acc_B = (-ang_vel_B[scene].cross(I_mat * ang_vel_B[scene])) + tau_B;
        ang_acc_B.x /= I[0]; ang_acc_B.y /= I[1]; ang_acc_B.z /= I[2];

        lin_vel_W[scene] += lin_acc_W * dt;
        lin_pos_W[scene] += lin_vel_W[scene] * dt;
        ang_vel_B[scene] += ang_acc_B * dt;
        R_W_B[scene] = R_W_B[scene] + (R_W_B[scene] * skew(ang_vel_B[scene])) * dt;

        model_matrices[scene][1] = Mat4::identity();
        model_matrices[scene][1].setTranslation(lin_pos_W[scene]);
        model_matrices[scene][1].setRotation(R_W_B[scene]);
        model_matrices[scene][1] = Mat4::scale(0.01f, 0.01f, 0.01f) * model_matrices[scene][1];
    }
}

int main() {
    const int width = 400, height = 300, num_objects = 2, num_scenes = 4, num_frames = 1000;
    
    std::vector<std::vector<Triangle>> triangles(num_objects);
    std::vector<unsigned char*> textures(num_objects);
    std::vector<int> tex_widths(num_objects), tex_heights(num_objects);

    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    Mat4 projection = create_projection_matrix(3.14159f / 4.0f, (float)width / height, 0.1f, 100.0f);

    Mat4 model_matrices[num_scenes][num_objects];
    for (int scene = 0; scene < num_scenes; scene++)
        for (int obj = 0; obj < num_objects; obj++)
            model_matrices[scene][obj] = create_model_matrix_random();

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

    Triangle *d_in_triangles, *d_out_triangles;
    int *d_offsets, *d_counts, *d_tex_widths, *d_tex_heights;
    unsigned char *d_textures, *d_output;
    float* d_zbuffer;
    Mat4* d_model_matrices;

    CHECK_CUDA(cudaMalloc(&d_in_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_out_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_offsets, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counts, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_textures, all_textures.size() * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, num_scenes * width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, num_scenes * width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_model_matrices, num_scenes * num_objects * sizeof(Mat4)));

    CHECK_CUDA(cudaMemcpy(d_in_triangles, all_triangles.data(), all_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, triangle_offsets.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_counts, triangle_counts.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_textures, all_textures.data(), all_textures.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_widths, all_tex_widths.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_heights, all_tex_heights.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));

    int max_triangles = *std::max_element(triangle_counts.begin(), triangle_counts.end());
    dim3 transform_block(256);
    dim3 transform_grid((max_triangles + transform_block.x - 1) / transform_block.x, num_scenes, num_objects);
    
    dim3 render_block(16, 16, 1);
    dim3 render_grid((width + render_block.x - 1) / render_block.x, 
                     (height + render_block.y - 1) / render_block.y, 
                     num_scenes);

    const float dt = 0.01f;

    std::vector<cv::VideoWriter> video_writers(num_scenes);
    for (int scene = 0; scene < num_scenes; scene++) {
        std::string filename = "output_scene" + std::to_string(scene) + ".mp4";
        video_writers[scene].open(filename, cv::VideoWriter::fourcc('a','v','c','1'), static_cast<int>(std::round(1.0f / dt)), cv::Size(width, height));
    }

    std::vector<float> omega(4, 50.01f);
    std::vector<Vec3> ang_vel_B(num_scenes), lin_vel_W(num_scenes), lin_pos_W(num_scenes);
    std::vector<Mat3> R_W_B(num_scenes, Mat3::identity());

    for (int scene = 0; scene < num_scenes; scene++) {
        lin_pos_W[scene] = Vec3(
            model_matrices[scene][1].m[3],
            model_matrices[scene][1].m[7],
            model_matrices[scene][1].m[11]
        );
    }

    std::vector<unsigned char> output(num_scenes * width * height * 3);
    for (int frame = 0; frame < num_frames; frame++) {
        for (int scene = 0; scene < num_scenes; scene++) {
            float rotation = 0.1f;
            Mat4 rotation_matrix = Mat4::rotationY(rotation);
            Vec3 translation(model_matrices[scene][0].m[3], 
                             model_matrices[scene][0].m[7], 
                             model_matrices[scene][0].m[11]);
            model_matrices[scene][0] = rotation_matrix * model_matrices[scene][0];
            model_matrices[scene][0].setTranslation(translation);
        }

        update_drone_dynamics(ang_vel_B, lin_vel_W, lin_pos_W, R_W_B, omega, model_matrices, dt);

        for (int scene = 0; scene < num_scenes; scene++) {
            Vec3 target_pos(1, 2, -5); // Set the target position for each scene
            drone_controller(ang_vel_B[scene], lin_vel_W[scene], lin_pos_W[scene], R_W_B[scene], target_pos, omega);
        }

        CHECK_CUDA(cudaMemcpy(d_model_matrices, model_matrices, num_scenes * num_objects * sizeof(Mat4), cudaMemcpyHostToDevice));

        transform_vertices_kernel<<<transform_grid, transform_block>>>(
            d_in_triangles, d_out_triangles, d_offsets, d_counts,
            d_model_matrices, projection, num_objects, num_scenes);

        render_kernel<<<render_grid, render_block>>>(
            d_out_triangles, d_offsets, d_counts,
            d_textures, d_tex_widths, d_tex_heights,
            d_output, d_zbuffer, width, height, num_objects, num_scenes);

        CHECK_CUDA(cudaMemcpy(output.data(), d_output, num_scenes * width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        for (int scene = 0; scene < num_scenes; scene++) {
            cv::Mat frame(height, width, CV_8UC3, output.data() + scene * width * height * 3);
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            video_writers[scene].write(frame);
        }
    }

    for (auto& writer : video_writers) {
        writer.release();
    }

    cudaFree(d_in_triangles);
    cudaFree(d_out_triangles);
    cudaFree(d_offsets);
    cudaFree(d_counts);
    cudaFree(d_textures);
    cudaFree(d_tex_widths);
    cudaFree(d_tex_heights);
    cudaFree(d_output);
    cudaFree(d_zbuffer);
    cudaFree(d_model_matrices);

    for (auto texture : textures) {
        stbi_image_free(texture);
    }

    return 0;
}