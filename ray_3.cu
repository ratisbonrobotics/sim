#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

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
        
        // Simple shadow check
        Ray shadow_ray(hit_point + normal * 0.001f, light_dir);
        for (int i = 0; i < sphere_count; i++) {
            float t;
            if (spheres[i].intersect(shadow_ray, t)) {
                diffuse *= 0.5f; // Soften shadows
                break;
            }
        }
        
        return hit_sphere->color * (diffuse * 0.7f + 0.2f); // Add some ambient light
    }

    Vec3 unit_direction = ray.direction.normalize();
    float t = 0.5f * (unit_direction.y + 1.0f);
    return Vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
}

__global__ void render(Vec3* fb, int width, int height, int samples, Sphere* spheres, int sphere_count, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
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
        col = col + color(r, spheres, sphere_count);
    }
    col = col * (1.0f / float(samples));
    fb[pixel_index] = col;
}

int main() {
    int width = 800;
    int height = 400;
    int samples = 4;
    int sphere_count = 2;

    Vec3* fb;
    CHECK_CUDA(cudaMallocManaged(&fb, width * height * sizeof(Vec3)));

    Sphere* spheres;
    CHECK_CUDA(cudaMallocManaged(&spheres, sphere_count * sizeof(Sphere)));
    spheres[0] = Sphere(Vec3(0, 0, -1), 0.5f, Vec3(0.7f, 0.3f, 0.3f));
    spheres[1] = Sphere(Vec3(0, -100.5f, -1), 100.0f, Vec3(0.3f, 0.7f, 0.3f));

    curandState* rand_state;
    CHECK_CUDA(cudaMalloc(&rand_state, width * height * sizeof(curandState)));

    dim3 blocks(width/16+1, height/16+1);
    dim3 threads(16, 16);

    render<<<blocks, threads>>>(fb, width, height, samples, spheres, sphere_count, rand_state);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("P3\n%d %d\n255\n", width, height);
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            int ir = int(255.99 * fb[pixel_index].x);
            int ig = int(255.99 * fb[pixel_index].y);
            int ib = int(255.99 * fb[pixel_index].z);
            printf("%d %d %d\n", ir, ig, ib);
        }
    }

    CHECK_CUDA(cudaFree(fb));
    CHECK_CUDA(cudaFree(spheres));
    CHECK_CUDA(cudaFree(rand_state));

    return 0;
}