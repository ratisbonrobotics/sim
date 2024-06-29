#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

const int WIDTH = 800;
const int HEIGHT = 600;

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    Vec3 normalize() const { float mag = std::sqrt(x * x + y * y + z * z); return Vec3(x / mag, y / mag, z / mag); }
    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    const float& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

struct Vec2 {
    float u, v;
    Vec2(float u = 0, float v = 0) : u(u), v(v) {}
    Vec2 operator+(const Vec2& other) const { return Vec2(u + other.u, v + other.v); }
    Vec2 operator/(float f) const { return Vec2(u / f, v / f); }
    Vec2 operator*(float f) const { return Vec2(u * f, v * f); }
    float& operator[](int i) { return i == 0 ? u : v; }
    const float& operator[](int i) const { return i == 0 ? u : v; }
};

Vec3 barycentric(const Vec2& A, const Vec2& B, const Vec2& C, const Vec2& P) {
    Vec3 s[2];
    for (int i = 2; i--; ) {
        s[i][0] = C[i] - A[i];
        s[i][1] = B[i] - A[i];
        s[i][2] = A[i] - P[i];
    }
    Vec3 u = s[0].cross(s[1]);
    if (std::abs(u.z) > 1e-2)
        return Vec3(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
    return Vec3(-1, 1, 1); // Triangle is degenerate
}

void triangle(Vec3* pts, Vec2* uvs, Vec3* normals, unsigned char* texture, int tex_width, int tex_height, unsigned char* framebuffer, float* zbuffer) {
    Vec2 bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2 bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    Vec2 clamp(WIDTH - 1, HEIGHT - 1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            bboxmin[j] = std::max(0.f, std::min(bboxmin[j], pts[i][j]));
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts[i][j]));
        }
    }
    Vec3 P;
    Vec3 light_dir = Vec3(0, 0, -1).normalize(); // Light coming from the front
    float ambient = 0.1f;
    
    for (P.x = bboxmin[0]; P.x <= bboxmax[0]; P.x++) {
        for (P.y = bboxmin[1]; P.y <= bboxmax[1]; P.y++) {
            Vec3 bc_screen = barycentric(Vec2(pts[0].x, pts[0].y), Vec2(pts[1].x, pts[1].y), Vec2(pts[2].x, pts[2].y), Vec2(P.x, P.y));
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
            
            // Debug output: Print barycentric coordinates
            std::cout << "Barycentric coords: " << bc_screen.x << ", " << bc_screen.y << ", " << bc_screen.z << std::endl;
            
            // Perspective-correct interpolation
            float Z = 0;
            Vec2 uv;
            Vec3 normal;
            for (int i = 0; i < 3; i++) {
                float w = bc_screen[i] / pts[i].z;
                Z += w;
                uv = uv + Vec2(uvs[i].u, uvs[i].v) * w;
                normal = normal + normals[i] * w;
            }
            uv = uv / Z;
            normal = normal.normalize();
            
            // Debug output: Print interpolated depth, texture coordinates, and normal
            std::cout << "Depth: " << Z << ", UV: " << uv.u << ", " << uv.v << ", Normal: " << normal.x << ", " << normal.y << ", " << normal.z << std::endl;
            
            float intensity = std::max(ambient, normal.dot(light_dir));
            
            int idx = int(P.x + P.y * WIDTH);
            if (zbuffer[idx] < Z) {
                zbuffer[idx] = Z;
                int tex_x = std::min(std::max(static_cast<int>(uv.u * tex_width), 0), tex_width - 1);
                int tex_y = std::min(std::max(static_cast<int>(uv.v * tex_height), 0), tex_height - 1);
                int tex_idx = (tex_x + tex_y * tex_width) * 3;
                int fb_idx = (static_cast<int>(P.x) + static_cast<int>(P.y) * WIDTH) * 3;
                for (int i = 0; i < 3; i++) {
                    framebuffer[fb_idx + i] = static_cast<unsigned char>(std::min(255.0f, texture[tex_idx + i] * intensity));
                }
            }
        }
    }
}

Vec3 calculate_normal(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 a = v1 - v0;
    Vec3 b = v2 - v0;
    return a.cross(b).normalize();
}

int main() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    const char* filename = "african_head.obj";
    const char* mtl_basedir = NULL;  // Use NULL if materials are in the same directory as the OBJ file
    bool triangulate = true;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename, mtl_basedir, triangulate);

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        std::cerr << "Failed to load/parse .obj file" << std::endl;
        return 1;
    }

    int tex_width, tex_height, tex_channels;
    unsigned char* texture = stbi_load("african_head_diffuse.tga", &tex_width, &tex_height, &tex_channels, 3);
    if (!texture) {
        std::cerr << "Failed to load texture" << std::endl;
        return 1;
    }

    std::vector<unsigned char> framebuffer(WIDTH * HEIGHT * 3, 0);
    std::vector<float> zbuffer(WIDTH * HEIGHT, -std::numeric_limits<float>::max());

    for (const auto& shape : shapes) {
        for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
            Vec3 world_coords[3];
            Vec2 uv[3];
            Vec3 normals[3];
            for (int i = 0; i < 3; i++) {
                tinyobj::index_t idx = shape.mesh.indices[3 * f + i];
                world_coords[i] = Vec3(attrib.vertices[3 * idx.vertex_index],
                                       attrib.vertices[3 * idx.vertex_index + 1],
                                       attrib.vertices[3 * idx.vertex_index + 2]);
                if (idx.texcoord_index >= 0) {
                    uv[i] = Vec2(attrib.texcoords[2 * idx.texcoord_index],
                                 1.f - attrib.texcoords[2 * idx.texcoord_index + 1]);
                }
                if (idx.normal_index >= 0) {
                    normals[i] = Vec3(attrib.normals[3 * idx.normal_index],
                                      attrib.normals[3 * idx.normal_index + 1],
                                      attrib.normals[3 * idx.normal_index + 2]);
                }
                
                // Transform vertices to screen space
                world_coords[i].x = (world_coords[i].x + 1.f) * WIDTH / 2.f;
                world_coords[i].y = (1.f - world_coords[i].y) * HEIGHT / 2.f;
            }
            
            // If normals are not provided in the OBJ file, calculate them
            if (attrib.normals.empty()) {
                Vec3 face_normal = calculate_normal(world_coords[0], world_coords[1], world_coords[2]);
                normals[0] = normals[1] = normals[2] = face_normal;
            }
            
            triangle(world_coords, uv, normals, texture, tex_width, tex_height, framebuffer.data(), zbuffer.data());
        }
    }

    stbi_write_png("output.png", WIDTH, HEIGHT, 3, framebuffer.data(), WIDTH * 3);

    stbi_image_free(texture);
    return 0;
}