#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include "tgaimage.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const int width  = 800;
const int height = 800;

struct Vec3f {
    float x, y, z;
    Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

Vec3f translate(const Vec3f &v, const Vec3f &t) {
    return Vec3f(v.x + t.x, v.y + t.y, v.z + t.z);
}

Vec3f perspectiveProject(const Vec3f &v, float distance) {
    float scale = distance / (distance + v.z);
    return Vec3f(v.x * scale, v.y * scale, v.z);
}

void drawLine(int x0, int y0, int x1, int y1, TGAImage &image, const TGAColor &color) {
    bool steep = false;
    if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    int derror2 = std::abs(dy) * 2;
    int error2 = 0;
    int y = y0;

    for (int x = x0; x <= x1; x++) {
        if (steep) {
            image.set(y, x, color);
        } else {
            image.set(x, y, color);
        }
        error2 += derror2;
        if (error2 > dx) {
            y += (y1 > y0 ? 1 : -1);
            error2 -= dx * 2;
        }
    }
}

void drawWireframe(const std::vector<Vec3f>& verts, const std::vector<std::vector<int>>& faces, TGAImage& image, const TGAColor& color, const Vec3f& position, float cameraDistance) {
    for (const auto& face : faces) {
        for (int j = 0; j < 3; j++) {
            Vec3f v0 = perspectiveProject(translate(verts[face[j]], position), cameraDistance);
            Vec3f v1 = perspectiveProject(translate(verts[face[(j + 1) % 3]], position), cameraDistance);

            int x0 = (v0.x + 1.0f) * width / 2.0f;
            int y0 = (v0.y + 1.0f) * height / 2.0f;
            int x1 = (v1.x + 1.0f) * width / 2.0f;
            int y1 = (v1.y + 1.0f) * height / 2.0f;
            drawLine(x0, y0, x1, y1, image, color);
        }
    }
}

bool loadObj(const std::string& filename, std::vector<Vec3f>& verts, std::vector<std::vector<int>>& faces) {
    std::ifstream inFile(filename);
    if (inFile.fail()) return false;

    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        char prefix;

        if (line.substr(0, 2) == "v ") {
            Vec3f vec;
            iss >> prefix >> vec.x >> vec.y >> vec.z;
            verts.push_back(vec);
        } else if (line.substr(0, 2) == "f ") {
            std::vector<int> face;
            int vIndex;
            iss >> prefix;
            while (iss >> vIndex) {
                face.push_back(vIndex - 1); // OBJ indices start at 1
                iss.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
            }
            faces.push_back(face);
        }
    }
    return true;
}

int main(int argc, char** argv) {
    TGAImage image(width, height, TGAImage::RGB);
    
    std::vector<Vec3f> vertices;
    std::vector<std::vector<int>> faces;

    if (!loadObj("african_head.obj", vertices, faces)) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return 1;
    }

    Vec3f position(0, 0, 30);
    float cameraDistance = 60.0f;

    drawWireframe(vertices, faces, image, white, position, cameraDistance);

    image.flip_vertically();
    image.write_tga_file("output.tga");
    return 0;
}