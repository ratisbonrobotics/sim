#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float d) const { return Vec3(x * d, y * d, z * d); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 normalize() const {
        float mg = sqrtf(x*x + y*y + z*z);
        return Vec3(x/mg, y/mg, z/mg);
    }
};

__host__ __device__ float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Ray {
    Vec3 origin, direction;
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    __host__ __device__ Sphere(const Vec3& center, float radius, const Vec3& color) 
        : center(center), radius(radius), color(color) {}
    
    __host__ __device__ bool intersect(const Ray& ray, float& t) const {
        Vec3 oc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return false;
        t = (-b - sqrtf(discriminant)) / (2.0f * a);
        return t > 0;
    }
};

__device__ Vec3 color(const Ray& ray, Sphere* spheres, int sphere_count) {
    float closest_t = INFINITY;
    Sphere* hit_sphere = nullptr;

    for (int i = 0; i < sphere_count; i++) {
        float t;
        if (spheres[i].intersect(ray, t) && t < closest_t) {
            closest_t = t;
            hit_sphere = &spheres[i];
        }
    }

    if (hit_sphere) {
        Vec3 hit_point = ray.origin + ray.direction * closest_t;
        Vec3 normal = (hit_point - hit_sphere->center).normalize();
        Vec3 light_dir = Vec3(1, 1, -1).normalize();
        float diffuse = fmaxf(0.0f, dot(normal, light_dir));
        
        Ray shadow_ray(hit_point + normal * 0.001f, light_dir);
        for (int i = 0; i < sphere_count; i++) {
            float t;
            if (spheres[i].intersect(shadow_ray, t)) {
                diffuse *= 0.5f;
                break;
            }
        }
        
        return hit_sphere->color * (diffuse * 0.7f + 0.2f);
    }

    Vec3 unit_direction = ray.direction.normalize();
    float t = 0.5f * (unit_direction.y + 1.0f);
    return Vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
}

__global__ void render(Vec3* fb, int width, int height, int samples, Sphere* spheres, int sphere_count, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int render_index = blockIdx.z;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = (render_index * width * height) + (j * width + i);
    curandState local_rand_state = rand_state[pixel_index];

    Vec3 lower_left_corner(-2.0f, -1.0f, -1.0f);
    Vec3 horizontal(4.0f, 0.0f, 0.0f);
    Vec3 vertical(0.0f, 2.0f, 0.0f);
    Vec3 origin(0.0f, 0.0f, 0.0f);

    Vec3 col(0, 0, 0);
    for (int s = 0; s < samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height);
        Ray r(origin, lower_left_corner + horizontal * u + vertical * v);
        col = col + color(r, spheres + render_index * sphere_count, sphere_count);
    }
    col = col * (1.0f / float(samples));
    fb[pixel_index] = col;
}

int main() {
    int width = 100;
    int height = 50;
    int samples = 4;
    int sphere_count = 2;

    // Array of scene counts to test
    int scene_counts[] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
    int num_tests = sizeof(scene_counts) / sizeof(scene_counts[0]);

    for (int test = 0; test < num_tests; test++) {
        int num_renders = scene_counts[test];

        Vec3* fb;
        CHECK_CUDA(cudaMallocManaged(&fb, num_renders * width * height * sizeof(Vec3)));

        Sphere* spheres;
        CHECK_CUDA(cudaMallocManaged(&spheres, num_renders * sphere_count * sizeof(Sphere)));

        srand(time(NULL));
        for (int r = 0; r < num_renders; r++) {
            float x = (float)rand() / RAND_MAX * 2 - 1;  // Random x between -1 and 1
            float y = (float)rand() / RAND_MAX * 2 - 1;  // Random y between -1 and 1
            spheres[r * sphere_count] = Sphere(Vec3(x, y, -1), 0.5f, Vec3(0.7f, 0.3f, 0.3f));
            spheres[r * sphere_count + 1] = Sphere(Vec3(0, -100.5f, -1), 100.0f, Vec3(0.3f, 0.7f, 0.3f));
        }

        curandState* rand_state;
        CHECK_CUDA(cudaMalloc(&rand_state, num_renders * width * height * sizeof(curandState)));

        dim3 blocks(width/16+1, height/16+1, num_renders);
        dim3 threads(16, 16);

        // Measure execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        render<<<blocks, threads>>>(fb, width, height, samples, spheres, sphere_count, rand_state);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Number of scenes: %d, Execution time: %.2f ms\n", num_renders, milliseconds);

        CHECK_CUDA(cudaFree(fb));
        CHECK_CUDA(cudaFree(spheres));
        CHECK_CUDA(cudaFree(rand_state));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}