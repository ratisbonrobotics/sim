#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <float.h> // For FLT_MAX

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3 operator+(const vec3& v) const { return vec3(x+v.x, y+v.y, z+v.z); }
    __host__ __device__ vec3 operator-(const vec3& v) const { return vec3(x-v.x, y-v.y, z-v.z); }
    __host__ __device__ vec3 operator*(float t) const { return vec3(x*t, y*t, z*t); }
    __host__ __device__ vec3 operator*(const vec3& v) const { return vec3(x*v.x, y*v.y, z*v.z); }
    __host__ __device__ vec3 operator/(float t) const { return vec3(x/t, y/t, z/t); }
};

__host__ __device__ vec3 operator*(float t, const vec3& v) {
    return v * t;
}

struct ray {
    vec3 origin, direction;
    __device__ ray(const vec3& a, const vec3& b) : origin(a), direction(b) {}
    __device__ vec3 point_at_parameter(float t) const { return origin + direction * t; }
};

__device__ float dot(const vec3& v1, const vec3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ vec3 unit_vector(const vec3& v) {
    float length = sqrt(dot(v, v));
    return v / length;
}

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1,1,1);
    } while (dot(p,p) >= 1.0f);
    return p;
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r, float t_min, float t_max, float& t, vec3& normal) {
    vec3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = dot(oc, r.direction);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            t = temp;
            normal = (r.point_at_parameter(t) - center) / radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            t = temp;
            normal = (r.point_at_parameter(t) - center) / radius;
            return true;
        }
    }
    return false;
}

__device__ vec3 color(const ray& r, curandState *local_rand_state) {
    vec3 cur_origin = r.origin;
    vec3 cur_direction = r.direction;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    for(int i = 0; i < 50; i++) {
        float t;
        vec3 normal;
        if (hit_sphere(vec3(0,0,-1), 0.5f, ray(cur_origin, cur_direction), 0.001f, FLT_MAX, t, normal) ||
            hit_sphere(vec3(0,-100.5f,-1), 100.0f, ray(cur_origin, cur_direction), 0.001f, FLT_MAX, t, normal) ||
            hit_sphere(vec3(1,0,-1), 0.5f, ray(cur_origin, cur_direction), 0.001f, FLT_MAX, t, normal) ||
            hit_sphere(vec3(-1,0,-1), 0.5f, ray(cur_origin, cur_direction), 0.001f, FLT_MAX, t, normal)) {
            vec3 target = cur_origin + cur_direction*t + normal + random_in_unit_sphere(local_rand_state);
            cur_origin = cur_origin + cur_direction*t;
            cur_direction = target - cur_origin;
            cur_attenuation = cur_attenuation * 0.5f;
        }
        else {
            vec3 unit_direction = unit_vector(cur_direction);
            float tt = 0.5f*(unit_direction.y + 1.0f);
            vec3 c = vec3(1.0f, 1.0f, 1.0f)*(1.0f-tt) + vec3(0.5f, 0.7f, 1.0f)*tt;
            return cur_attenuation * c;
        }
    }
    return vec3(0,0,0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r(vec3(-2,2,1), vec3(4*u-2, -4*v+2, -2));
        col = col + color(r, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col = col / float(ns);
    col = vec3(sqrt(col.x), sqrt(col.y), sqrt(col.z));
    fb[pixel_index] = col;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].x);
            int ig = int(255.99*fb[pixel_index].y);
            int ib = int(255.99*fb[pixel_index].z);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));

    return 0;
}