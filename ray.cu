#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

struct vec3 {
    float x = 0, y = 0, z = 0;

    __host__ __device__ float& operator[](const int i) { return i == 0 ? x : (1 == i ? y : z); }
    __host__ __device__ const float& operator[](const int i) const { return i == 0 ? x : (1 == i ? y : z); }

    __host__ __device__ vec3 operator*(const float v) const { return { x * v, y * v, z * v }; }
    __host__ __device__ float operator*(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ vec3 operator+(const vec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
    __host__ __device__ vec3 operator-(const vec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
    __host__ __device__ vec3 operator-() const { return { -x, -y, -z }; }

    __host__ __device__ float norm() const { return std::sqrt(x * x + y * y + z * z); }
    __host__ __device__ vec3 normalized() const { return (*this) * (1.f / norm()); }
};

__host__ __device__ vec3 cross(const vec3& v1, const vec3& v2) {
    return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
}

struct Material {
    float refractive_index = 1;
    float albedo[4] = { 2,0,0,0 };
    vec3 diffuse_color = { 0,0,0 };
    float specular_exponent = 0;
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

__constant__ Material ivory;
__constant__ Material glass;
__constant__ Material red_rubber;
__constant__ Material mirror;

__constant__ Sphere spheres[4];
__constant__ vec3 lights[3];

__device__ vec3 reflect(const vec3& I, const vec3& N) {
    return I - N * 2.f * (I * N);
}

__device__ vec3 refract(const vec3& I, const vec3& N, const float eta_t, const float eta_i = 1.f) { // Snell's law
    float cosi = -fmaxf(-1.f, fminf(1.f, I * N));
    if (cosi < 0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? vec3{ 1,0,0 } : I * eta + N * (eta * cosi - std::sqrt(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

struct Intersection {
    bool hit;
    float dist;
    vec3 point;
    vec3 N;
    Material material;
};

__device__ Intersection ray_sphere_intersect(const vec3& orig, const vec3& dir, const Sphere& s) {
    Intersection result;
    vec3 L = s.center - orig;
    float tca = L * dir;
    float d2 = L * L - tca * tca;
    if (d2 > s.radius * s.radius) {
        result.hit = false;
        return result;
    }
    float thc = std::sqrt(s.radius * s.radius - d2);
    float t0 = tca - thc, t1 = tca + thc;
    if (t0 > .001) {
        result.hit = true;
        result.dist = t0;
        return result;
    }
    if (t1 > .001) {
        result.hit = true;
        result.dist = t1;
        return result;
    }
    result.hit = false;
    return result;
}

__device__ Intersection scene_intersect(const vec3& orig, const vec3& dir) {
    Intersection result;
    result.dist = 1e10;
    result.hit = false;
    if (std::abs(dir.y) > .001) { // intersect the ray with the checkerboard, avoid division by zero
        float d = -(orig.y + 4) / dir.y; // the checkerboard plane has equation y = -4
        vec3 p = orig + dir * d;
        if (d > .001 && d < result.dist && std::abs(p.x) < 10 && p.z < -10 && p.z > -30) {
            result.dist = d;
            result.point = p;
            result.N = { 0,1,0 };
            result.material.diffuse_color = (int(.5 * result.point.x + 1000) + int(.5 * result.point.z)) & 1 ? vec3{ .3, .3, .3 } : vec3{ .3, .2, .1 };
            result.hit = true;
        }
    }

    for (int i = 0; i < 4; i++) { // intersect the ray with all spheres
        const Sphere& s = spheres[i];
        Intersection sph_inter = ray_sphere_intersect(orig, dir, s);
        if (!sph_inter.hit || sph_inter.dist > result.dist) continue;
        result = sph_inter;
        result.point = orig + dir * result.dist;
        result.N = (result.point - s.center).normalized();
        result.material = s.material;
        result.hit = true;
    }
    return result;
}

__device__ vec3 cast_ray(const vec3& orig, const vec3& dir, const int depth = 0) {
    Intersection inter = scene_intersect(orig, dir);
    if (depth > 4 || !inter.hit)
        return { 0.2, 0.7, 0.8 }; // background color

    vec3 reflect_dir = reflect(dir, inter.N).normalized();
    vec3 refract_dir = refract(dir, inter.N, inter.material.refractive_index).normalized();
    vec3 reflect_color = cast_ray(inter.point, reflect_dir, depth + 1);
    vec3 refract_color = cast_ray(inter.point, refract_dir, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (int i = 0; i < 3; i++) { // checking if the point lies in the shadow of the light
        const vec3& light = lights[i];
        vec3 light_dir = (light - inter.point).normalized();
        Intersection shadow_inter = scene_intersect(inter.point, light_dir);
        if (shadow_inter.hit && (shadow_inter.point - inter.point).norm() < (light - inter.point).norm()) continue;
        diffuse_light_intensity += fmaxf(0.f, light_dir * inter.N);
        specular_light_intensity += powf(fmaxf(0.f, -reflect(-light_dir, inter.N) * dir), inter.material.specular_exponent);
    }

    return inter.material.diffuse_color * diffuse_light_intensity * inter.material.albedo[0] + vec3{ 1., 1., 1. }*specular_light_intensity * inter.material.albedo[1] + reflect_color*inter.material.albedo[2] + refract_color*inter.material.albedo[3];
}

__global__ void render_kernel(vec3* framebuffer, int width, int height, float fov) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pix = y * width + x;
    float dir_x = (x + 0.5f) - width / 2.0f;
    float dir_y = -(y + 0.5f) + height / 2.0f; // this flips the image at the same time
    float dir_z = -height / (2.0f * tan(fov / 2.0f));
    framebuffer[pix] = cast_ray(vec3{ 0,0,0 }, vec3{ dir_x, dir_y, dir_z }.normalized());
}

void save_image(const vec3* framebuffer, int width, int height) {
    std::ofstream ofs("./out.ppm", std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        vec3 color = framebuffer[i];
        float max = std::max(1.f, std::max(color[0], std::max(color[1], color[2])));
        for (const int chan : { 0,1,2 })
            ofs << (char)(255 * color[chan] / max);
    }
}

int main() {
    constexpr int width = 1024;
    constexpr int height = 768;
    constexpr float fov = 1.05f; // 60 degrees field of view in radians

    // Define materials
    Material h_ivory = { 1.0, {0.9f,  0.5f, 0.1f, 0.0f}, {0.4f, 0.4f, 0.3f}, 50.0f };
    Material h_glass = { 1.5, {0.0f,  0.9f, 0.1f, 0.8f}, {0.6f, 0.7f, 0.8f}, 125.0f };
    Material h_red_rubber = { 1.0, {1.4f,  0.3f, 0.0f, 0.0f}, {0.3f, 0.1f, 0.1f}, 10.0f };
    Material h_mirror = { 1.0, {0.0f, 16.0f, 0.8f, 0.0f}, {1.0f, 1.0f, 1.0f}, 1425.0f };

    // Define spheres
    Sphere h_spheres[4] = {
        {{-3.0f,    0.0f,   -16.0f}, 2.0f,      h_ivory},
        {{-1.0f, -1.5f, -12.0f}, 2.0f,      h_glass},
        {{ 1.5f, -0.5f, -18.0f}, 3.0f, h_red_rubber},
        {{ 7.0f,    5.0f,   -18.0f}, 4.0f,     h_mirror}
    };

    // Define lights
    vec3 h_lights[3] = {
        {-20.0f, 20.0f,  20.0f},
        { 30.0f, 50.0f, -25.0f},
        { 30.0f, 20.0f,  30.0f}
    };

    // Copy materials to constant memory
    cudaMemcpyToSymbol(ivory, &h_ivory, sizeof(Material));
    cudaMemcpyToSymbol(glass, &h_glass, sizeof(Material));
    cudaMemcpyToSymbol(red_rubber, &h_red_rubber, sizeof(Material));
    cudaMemcpyToSymbol(mirror, &h_mirror, sizeof(Material));

    // Copy spheres to constant memory
    cudaMemcpyToSymbol(spheres, h_spheres, sizeof(Sphere) * 4);

    // Copy lights to constant memory
    cudaMemcpyToSymbol(lights, h_lights, sizeof(vec3) * 3);

    vec3* d_framebuffer;
    cudaMalloc(&d_framebuffer, width * height * sizeof(vec3));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_kernel <<<numBlocks, threadsPerBlock>>> (d_framebuffer, width, height, fov);
    cudaDeviceSynchronize();

    vec3* h_framebuffer = new vec3[width * height];
    cudaMemcpy(h_framebuffer, d_framebuffer, width * height * sizeof(vec3), cudaMemcpyDeviceToHost);

    save_image(h_framebuffer, width, height);

    delete[] h_framebuffer;
    cudaFree(d_framebuffer);
    return 0;
}