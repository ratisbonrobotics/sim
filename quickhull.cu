#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>

using namespace std;

// Structure to hold points in 3D space
struct Point {
    float x, y, z;
};

__device__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 subtract(const Point &a, const Point &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__global__ void find_furthest_points(Point *d_points, int num_points, Point *d_extreme_points) {
    // Example kernel to find furthest points (incomplete)
    // Needs implementation to find actual furthest points in 3D
    // Placeholder Example with the first point
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        // For simplicity, just set the first 4 points as "extreme"
        if (idx < 4) {
            d_extreme_points[idx] = d_points[idx];
        }
    }
}

vector<Point> read_points_from_stdin() {
    vector<Point> points;
    string line;
    
    // Skip the first two lines
    getline(cin, line); // "3 rbox D3"
    getline(cin, line); // "50"
    
    while (getline(cin, line)) {
        stringstream ss(line);
        Point p;
        ss >> p.x >> p.y >> p.z;
        points.push_back(p);
    }
    
    return points;
}

void quickhull(const vector<Point> &points) {
    int num_points = points.size();
    Point *d_points;
    cudaMalloc(&d_points, num_points * sizeof(Point));
    cudaMemcpy(d_points, points.data(), num_points * sizeof(Point), cudaMemcpyHostToDevice);

    // Extreme points - allocating space for 4 points
    const int NUM_EXTREME_POINTS = 4;
    Point *d_extreme_points;
    cudaMalloc(&d_extreme_points, NUM_EXTREME_POINTS * sizeof(Point));

    // Kernel invocation - finding furthest points (placeholder example)
    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    find_furthest_points<<<numBlocks, blockSize>>>(d_points, num_points, d_extreme_points);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy extreme points back to host
    vector<Point> extreme_points(NUM_EXTREME_POINTS);
    cudaMemcpy(extreme_points.data(), d_extreme_points, NUM_EXTREME_POINTS * sizeof(Point), cudaMemcpyDeviceToHost);

    // Print the extreme points
    for (const auto &p : extreme_points) {
        cout << p.x << " " << p.y << " " << p.z << endl;
    }

    // Clean up
    cudaFree(d_points);
    cudaFree(d_extreme_points);
}

int main() {
    vector<Point> points = read_points_from_stdin();
    quickhull(points);
    return 0;
}