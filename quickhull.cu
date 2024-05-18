#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

struct Point {
    double x, y, z;
};

__device__ double cross(const Point& a, const Point& b, const Point& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__ double dist(const Point& p, const Point& a, const Point& b, const Point& c) {
    Point ab = {b.x - a.x, b.y - a.y, b.z - a.z};
    Point ac = {c.x - a.x, c.y - a.y, c.z - a.z};
    Point ap = {p.x - a.x, p.y - a.y, p.z - a.z};

    Point n = {ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z, ab.x * ac.y - ab.y * ac.x};
    double d = n.x * ap.x + n.y * ap.y + n.z * ap.z;
    double len = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

    return fabs(d) / len;
}

__global__ void quickhull_kernel(Point* points, int num_points, Point* hull, int* hull_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point p = points[idx];
    double max_dist = 0;
    int max_idx = -1;

    for (int i = 0; i < *hull_size; i += 3) {
        double d = dist(p, hull[i], hull[i + 1], hull[i + 2]);
        if (d > max_dist) {
            max_dist = d;
            max_idx = i;
        }
    }

    if (max_idx != -1) {
        int new_size = atomicAdd(hull_size, 3);
        hull[new_size] = p;
        hull[new_size + 1] = hull[max_idx + 1];
        hull[new_size + 2] = hull[max_idx + 2];
        hull[max_idx + 1] = p;
    }
}

void quickhull(vector<Point>& points, vector<Point>& hull) {
    int num_points = points.size();
    Point* d_points;
    cudaMalloc(&d_points, num_points * sizeof(Point));
    cudaMemcpy(d_points, points.data(), num_points * sizeof(Point), cudaMemcpyHostToDevice);

    Point* d_hull;
    cudaMalloc(&d_hull, num_points * sizeof(Point));

    int* d_hull_size;
    cudaMalloc(&d_hull_size, sizeof(int));
    cudaMemset(d_hull_size, 0, sizeof(int));

    int block_size = 256;
    int num_blocks = (num_points + block_size - 1) / block_size;

    quickhull_kernel<<<num_blocks, block_size>>>(d_points, num_points, d_hull, d_hull_size);
    cudaDeviceSynchronize();

    int hull_size;
    cudaMemcpy(&hull_size, d_hull_size, sizeof(int), cudaMemcpyDeviceToHost);
    hull.resize(hull_size);
    cudaMemcpy(hull.data(), d_hull, hull_size * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_hull);
    cudaFree(d_hull_size);
}

int main() {
    vector<Point> points;
    double x, y, z;
    while (cin >> x >> y >> z) {
        points.push_back({x, y, z});
    }

    vector<Point> hull;
    quickhull(points, hull);

    for (const auto& p : hull) {
        cout << p.x << " " << p.y << " " << p.z << endl;
    }

    return 0;
}