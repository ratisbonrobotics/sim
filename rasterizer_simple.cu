#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

struct Vec3f { float x, y, z; };
struct Vec2f { float u, v; };
struct Triangle { Vec3f v[3], n[3]; Vec2f uv[3]; };
struct Mat4f { float m[4][4]; };

__device__ Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f s[2] = {{C.x-A.x, B.x-A.x, A.x-P.x}, {C.y-A.y, B.y-A.y, A.y-P.y}};
    Vec3f u = {s[0].y*s[1].z - s[0].z*s[1].y, s[0].z*s[1].x - s[0].x*s[1].z, s[0].x*s[1].y - s[0].y*s[1].x};
    return fabs(u.z) > 1e-2 ? Vec3f{1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z} : Vec3f{-1,1,1};
}

__host__ __device__ Vec3f mat_transform(const Mat4f& m, const Vec3f& v) {
    float w = m.m[3][0]*v.x + m.m[3][1]*v.y + m.m[3][2]*v.z + m.m[3][3];
    return {(m.m[0][0]*v.x + m.m[0][1]*v.y + m.m[0][2]*v.z + m.m[0][3])/w,
            (m.m[1][0]*v.x + m.m[1][1]*v.y + m.m[1][2]*v.z + m.m[1][3])/w,
            (m.m[2][0]*v.x + m.m[2][1]*v.y + m.m[2][2]*v.z + m.m[2][3])/w};
}

__global__ void rasterize_kernel(Vec3f* vertices, Triangle* triangles, int* triangle_counts, 
                                 unsigned char* textures, int* tex_widths, int* tex_heights, 
                                 Mat4f* model_matrices, unsigned char* output, float* zbuffer, 
                                 int width, int height, int num_objects) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (height-1-y) * width + x;
    zbuffer[idx] = FLT_MAX;
    Vec3f color = {0.2f, 0.2f, 0.2f};
    Vec3f light_dir = {0.577f, 0.577f, 0.577f};  // Normalized direction
    Vec3f P = {(float)x, (float)(height-1-y), 0};

    int vertex_offset = 0, triangle_offset = 0, texture_offset = 0;
    for (int obj = 0; obj < num_objects; obj++) {
        for (int i = 0; i < triangle_counts[obj]; i++) {
            Triangle& tri = triangles[triangle_offset + i];
            Vec3f screen_coords[3];
            for (int j = 0; j < 3; j++) {
                Vec3f v = vertices[vertex_offset + i*3 + j];
                screen_coords[j] = {(v.x/v.z+1.0f)*width/2.0f, height-1-(v.y/v.z+1.0f)*height/2.0f, v.z};
            }

            Vec3f bc_screen = barycentric(screen_coords[0], screen_coords[1], screen_coords[2], P);
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

            float frag_depth = bc_screen.x*screen_coords[0].z + bc_screen.y*screen_coords[1].z + bc_screen.z*screen_coords[2].z;
            if (frag_depth < zbuffer[idx]) {
                zbuffer[idx] = frag_depth;
                Vec2f uv = {tri.uv[0].u*bc_screen.x + tri.uv[1].u*bc_screen.y + tri.uv[2].u*bc_screen.z,
                            tri.uv[0].v*bc_screen.x + tri.uv[1].v*bc_screen.y + tri.uv[2].v*bc_screen.z};
                int tex_x = uv.u * tex_widths[obj];
                int tex_y = (1.0f - uv.v) * tex_heights[obj];
                int tex_idx = texture_offset + (tex_y * tex_widths[obj] + tex_x) * 3;
                Vec3f tex_color = {textures[tex_idx]/255.0f, textures[tex_idx+1]/255.0f, textures[tex_idx+2]/255.0f};
                Vec3f normal = {tri.n[0].x*bc_screen.x + tri.n[1].x*bc_screen.y + tri.n[2].x*bc_screen.z,
                                tri.n[0].y*bc_screen.x + tri.n[1].y*bc_screen.y + tri.n[2].y*bc_screen.z,
                                tri.n[0].z*bc_screen.x + tri.n[1].z*bc_screen.y + tri.n[2].z*bc_screen.z};
                float len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
                normal = {normal.x/len, normal.y/len, normal.z/len};
                normal = mat_transform(model_matrices[obj], normal);
                
                // Improved lighting calculation
                float ambient = 0.3f;
                float diffuse = fmaxf(0.0f, normal.x*light_dir.x + normal.y*light_dir.y + normal.z*light_dir.z);
                float specular = powf(fmaxf(0.0f, 2.0f*diffuse*normal.z - light_dir.z), 32.0f);
                
                color.x = fminf(1.0f, tex_color.x * (ambient + 0.7f*diffuse) + 0.3f*specular);
                color.y = fminf(1.0f, tex_color.y * (ambient + 0.7f*diffuse) + 0.3f*specular);
                color.z = fminf(1.0f, tex_color.z * (ambient + 0.7f*diffuse) + 0.3f*specular);
            }
        }
        vertex_offset += triangle_counts[obj] * 3;
        triangle_offset += triangle_counts[obj];
        texture_offset += tex_widths[obj] * tex_heights[obj] * 3;
    }

    output[idx*3] = fminf(color.x*255.0f, 255.0f);
    output[idx*3+1] = fminf(color.y*255.0f, 255.0f);
    output[idx*3+2] = fminf(color.z*255.0f, 255.0f);
}

void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) { printf("Failed to open OBJ file: %s\n", filename); return; }

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
                tri.v[i] = vertices[v-1];
                tri.uv[i] = texcoords[vt-1];
                tri.n[i] = normals[vn-1];
            }
            triangles.push_back(tri);
        }
    }
}

int main() {
    const int width = 800, height = 600, num_objects = 2;
    std::vector<Triangle> triangles[num_objects];
    unsigned char* textures[num_objects];
    int tex_widths[num_objects], tex_heights[num_objects];
    
    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    for (int i = 0; i < num_objects; i++)
        if (!textures[i]) { printf("Failed to load texture for object %d\n", i); return 1; }

    Mat4f model_matrices[num_objects] = {{{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}}, {{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}}};
    model_matrices[0].m[0][3] = -1.0f; model_matrices[0].m[2][3] = -3.0f;
    float angle = 3.14159f / 4.0f;
    model_matrices[0].m[0][0] = model_matrices[0].m[2][2] = cos(angle);
    model_matrices[0].m[0][2] = -(model_matrices[0].m[2][0] = -sin(angle));
    model_matrices[1].m[0][3] = 1.0f; model_matrices[1].m[1][3] = 0.5f; model_matrices[1].m[2][3] = -2.5f;
    model_matrices[1].m[0][0] = model_matrices[1].m[1][1] = model_matrices[1].m[2][2] = 0.1f;

    Mat4f proj = {{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,1,0}};
    float f = 1.0f / tanf(3.14159f / 8.0f);
    proj.m[0][0] = f / ((float)width / height);
    proj.m[1][1] = f;
    proj.m[2][2] = -1.1f;
    proj.m[2][3] = -0.1f;
    proj.m[3][2] = -1.0f;

    std::vector<Vec3f> projected_vertices;
    for (int i = 0; i < num_objects; i++)
        for (const auto& tri : triangles[i])
            for (int j = 0; j < 3; j++)
                projected_vertices.push_back(mat_transform(proj, mat_transform(model_matrices[i], tri.v[j])));

    Triangle* d_triangles;
    unsigned char* d_textures;
    int* d_triangle_counts, *d_tex_widths, *d_tex_heights;
    Mat4f* d_model_matrices;
    Vec3f* d_projected_vertices;
    unsigned char* d_output;
    float* d_zbuffer;

    int total_triangles = triangles[0].size() + triangles[1].size();
    int total_texture_size = (tex_widths[0] * tex_heights[0] + tex_widths[1] * tex_heights[1]) * 3;

    CHECK_CUDA(cudaMalloc(&d_triangles, total_triangles * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_textures, total_texture_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_triangle_counts, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_model_matrices, num_objects * sizeof(Mat4f)));
    CHECK_CUDA(cudaMalloc(&d_projected_vertices, projected_vertices.size() * sizeof(Vec3f)));
    CHECK_CUDA(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, width * height * sizeof(float)));

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

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    rasterize_kernel<<<grid_size, block_size>>>(d_projected_vertices, d_triangles, d_triangle_counts, 
                                                d_textures, d_tex_widths, d_tex_heights, 
                                                d_model_matrices, d_output, d_zbuffer, width, height, num_objects);

    unsigned char* output = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_png("output.png", width, height, 3, output, width * 3);

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