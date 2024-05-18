#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

struct Point {
    double x, y, z;

    Point operator-(const Point& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
};

struct Face {
    Point p[3];
    Point normal;
};

double dot(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Point cross(const Point& a, const Point& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

void quickhull(const std::vector<Point>& points, std::vector<Face>& faces) {
    if (points.size() < 4) return;

    // Find the initial tetrahedron
    int i0 = 0, i1 = 1, i2 = 2, i3 = 3;
    for (int i = 1; i < points.size(); ++i) {
        if (points[i].x < points[i0].x) i0 = i;
        if (points[i].x > points[i1].x) i1 = i;
        if (points[i].y < points[i2].y) i2 = i;
        if (points[i].z < points[i3].z) i3 = i;
    }

    Face face0 = {{points[i0], points[i1], points[i2]}, cross(points[i1] - points[i0], points[i2] - points[i0])};
    Face face1 = {{points[i0], points[i2], points[i3]}, cross(points[i2] - points[i0], points[i3] - points[i0])};
    Face face2 = {{points[i0], points[i3], points[i1]}, cross(points[i3] - points[i0], points[i1] - points[i0])};
    Face face3 = {{points[i1], points[i3], points[i2]}, cross(points[i3] - points[i1], points[i2] - points[i1])};

    faces.push_back(face0);
    faces.push_back(face1);
    faces.push_back(face2);
    faces.push_back(face3);

    // Assign points to faces
    std::vector<std::vector<int>> outside(4);
    for (int i = 0; i < points.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            if (dot(points[i] - faces[j].p[0], faces[j].normal) > 0) {
                outside[j].push_back(i);
            }
        }
    }

    // Recursively build the convex hull
    for (int i = 0; i < 4; ++i) {
        if (!outside[i].empty()) {
            Point furthest = points[outside[i][0]];
            double max_dist = dot(furthest - faces[i].p[0], faces[i].normal);
            for (int j = 1; j < outside[i].size(); ++j) {
                double dist = dot(points[outside[i][j]] - faces[i].p[0], faces[i].normal);
                if (dist > max_dist) {
                    furthest = points[outside[i][j]];
                    max_dist = dist;
                }
            }

            std::vector<Face> new_faces;
            for (int j = 0; j < 3; ++j) {
                Face new_face = {{faces[i].p[j], faces[i].p[(j + 1) % 3], furthest},
                                 cross(faces[i].p[(j + 1) % 3] - faces[i].p[j], furthest - faces[i].p[j])};
                new_faces.push_back(new_face);
            }

            std::vector<std::vector<int>> new_outside(3);
            for (int j = 0; j < outside[i].size(); ++j) {
                for (int k = 0; k < 3; ++k) {
                    if (dot(points[outside[i][j]] - new_faces[k].p[0], new_faces[k].normal) > 0) {
                        new_outside[k].push_back(outside[i][j]);
                    }
                }
            }

            faces.erase(faces.begin() + i);
            faces.insert(faces.end(), new_faces.begin(), new_faces.end());
            quickhull(points, faces);
        }
    }
}

int main() {
    std::ifstream input("rbox_verts.txt");
    int dim, num_points;
    input >> dim >> num_points;

    std::vector<Point> points(num_points);
    for (int i = 0; i < num_points; ++i) {
        input >> points[i].x >> points[i].y >> points[i].z;
    }

    std::vector<Face> faces;
    quickhull(points, faces);

    std::cout << "Convex Hull Faces:" << std::endl;
    for (const auto& face : faces) {
        std::cout << "(" << face.p[0].x << ", " << face.p[0].y << ", " << face.p[0].z << ") "
                  << "(" << face.p[1].x << ", " << face.p[1].y << ", " << face.p[1].z << ") "
                  << "(" << face.p[2].x << ", " << face.p[2].y << ", " << face.p[2].z << ")" << std::endl;
    }

    return 0;
}