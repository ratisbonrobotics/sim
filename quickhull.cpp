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
};

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

    // Print the initial points of the convex hull
    std::cout << "Initial points of the convex hull:" << std::endl;
    std::cout << minPoint.x << " " << minPoint.y << " " << minPoint.z << std::endl;
    std::cout << maxPoint.x << " " << maxPoint.y << " " << maxPoint.z << std::endl;
    std::cout << maxDistPoint1.x << " " << maxDistPoint1.y << " " << maxDistPoint1.z << std::endl;
    std::cout << maxDistPoint2.x << " " << maxDistPoint2.y << " " << maxDistPoint2.z << std::endl;

    return 0;
}