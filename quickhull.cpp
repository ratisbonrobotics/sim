#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

struct Point {
    double x, y, z;

    Point operator-(const Point& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Point cross(const Point& other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }

    double dot(const Point& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};

struct Face {
    Point a, b, c;
};

double distanceToFace(const Point& p, const Face& f) {
    Point normal = (f.b - f.a).cross(f.c - f.a);
    double d = -normal.dot(f.a);
    return std::abs(normal.dot(p) + d) / normal.length();
}


int main() {
    std::ifstream file("rbox_verts.txt");
    if (!file) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::vector<Point> points;
    std::string line;

    // Skip the first two lines
    std::getline(file, line);
    std::getline(file, line);

    while (std::getline(file, line)) {
        Point point;
        if (std::sscanf(line.c_str(), "%lf %lf %lf", &point.x, &point.y, &point.z) == 3) {
            points.push_back(point);
        }
    }

    file.close();

    // Find the points with minimum and maximum coordinates
    Point minPoint = points[0];
    Point maxPoint = points[0];
    for (const auto& point : points) {
        minPoint.x = std::min(minPoint.x, point.x);
        minPoint.y = std::min(minPoint.y, point.y);
        minPoint.z = std::min(minPoint.z, point.z);
        maxPoint.x = std::max(maxPoint.x, point.x);
        maxPoint.y = std::max(maxPoint.y, point.y);
        maxPoint.z = std::max(maxPoint.z, point.z);
    }

    // Find the points with maximum distance from the min and max points
    Point maxDistPoint1 = points[0];
    Point maxDistPoint2 = points[0];
    double maxDist1 = 0.0;
    double maxDist2 = 0.0;
    for (const auto& point : points) {
        double dist1 = std::sqrt(std::pow(point.x - minPoint.x, 2) +
                                 std::pow(point.y - minPoint.y, 2) +
                                 std::pow(point.z - minPoint.z, 2));
        double dist2 = std::sqrt(std::pow(point.x - maxPoint.x, 2) +
                                 std::pow(point.y - maxPoint.y, 2) +
                                 std::pow(point.z - maxPoint.z, 2));
        if (dist1 > maxDist1) {
            maxDist1 = dist1;
            maxDistPoint1 = point;
        }
        if (dist2 > maxDist2) {
            maxDist2 = dist2;
            maxDistPoint2 = point;
        }
    }

    // Create the initial tetrahedron
    std::vector<Face> faces = {
        {minPoint, maxDistPoint1, maxDistPoint2},
        {maxPoint, maxDistPoint1, maxDistPoint2},
        {minPoint, maxPoint, maxDistPoint1},
        {minPoint, maxPoint, maxDistPoint2}
    };

    // Find the external point for each face
    for (const auto& face : faces) {
        Point externalPoint = points[0];
        double maxDistance = 0.0;
        for (const auto& point : points) {
            double distance = distanceToFace(point, face);
            if (distance > maxDistance) {
                maxDistance = distance;
                externalPoint = point;
            }
        }
        std::cout << "External point for face (" << face.a.x << ", " << face.a.y << ", " << face.a.z << "), ("
                  << face.b.x << ", " << face.b.y << ", " << face.b.z << "), ("
                  << face.c.x << ", " << face.c.y << ", " << face.c.z << "): "
                  << externalPoint.x << " " << externalPoint.y << " " << externalPoint.z << std::endl;
    }

    return 0;
}