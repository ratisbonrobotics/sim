#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include "tgaimage.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const int width  = 800;
const int height = 800;

class Vec3f {
public:
    float x, y, z;
    Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

class Model {
private:
    std::vector<Vec3f> verts_;
    std::vector<std::vector<int>> faces_;
public:
    Model(const char *filename) {
        std::ifstream in;
        in.open(filename, std::ifstream::in);
        if (in.fail()) return;
        std::string line;
        while (!in.eof()) {
            std::getline(in, line);
            std::istringstream iss(line.c_str());
            char trash;
            if (!line.compare(0, 2, "v ")) {
                iss >> trash;
                Vec3f v;
                iss >> v.x >> v.y >> v.z;
                verts_.push_back(v);
            } else if (!line.compare(0, 2, "f ")) {
                std::vector<int> f;
                int itrash, idx;
                iss >> trash;
                while (iss >> idx >> trash >> itrash >> trash >> itrash) {
                    idx--; // in wavefront obj all indices start at 1, not zero
                    f.push_back(idx);
                }
                faces_.push_back(f);
            }
        }
        std::cerr << "# v# " << verts_.size() << " f# "  << faces_.size() << std::endl;
    }
    int nverts() { return (int)verts_.size(); }
    int nfaces() { return (int)faces_.size(); }
    std::vector<int> face(int idx) { return faces_[idx]; }
    Vec3f vert(int i) { return verts_[i]; }
};

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    bool steep = false;
    if (std::abs(x0-x1)<std::abs(y0-y1)) {
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0>x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx = x1-x0;
    int dy = y1-y0;
    int derror2 = std::abs(dy)*2;
    int error2 = 0;
    int y = y0;
    for (int x=x0; x<=x1; x++) {
        if (steep) {
            image.set(y, x, color);
        } else {
            image.set(x, y, color);
        }
        error2 += derror2;
        if (error2 > dx) {
            y += (y1>y0?1:-1);
            error2 -= dx*2;
        }
    }
}

int main(int argc, char** argv) {
    TGAImage image(width, height, TGAImage::RGB);
    Model model("african_head.obj");

    for (int i=0; i<model.nfaces(); i++) {
        std::vector<int> face = model.face(i);
        for (int j=0; j<3; j++) {
            Vec3f v0 = model.vert(face[j]);
            Vec3f v1 = model.vert(face[(j+1)%3]);
            int x0 = (v0.x+1.)*width/2.;
            int y0 = (v0.y+1.)*height/2.;
            int x1 = (v1.x+1.)*width/2.;
            int y1 = (v1.y+1.)*height/2.;
            line(x0, y0, x1, y1, image, white);
        }
    }

    image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    image.write_tga_file("output.tga");
    return 0;
}