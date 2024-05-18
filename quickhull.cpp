#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

    // Print the points (optional)
    for (const auto& point : points) {
        std::cout << point.x << " " << point.y << " " << point.z << std::endl;
    }

    return 0;
}