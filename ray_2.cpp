#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

class Vec3 {
public:
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double d) const { return Vec3(x * d, y * d, z * d); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 normalize() const {
        double mg = sqrt(x*x + y*y + z*z);
        return Vec3(x/mg, y/mg, z/mg);
    }
};

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

class Ray {
public:
    Vec3 origin, direction;
    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
};

class Sphere {
public:
    Vec3 center;
    double radius;
    Vec3 color;
    Sphere(const Vec3& center, double radius, const Vec3& color) 
        : center(center), radius(radius), color(color) {}
    
    bool intersect(const Ray& ray, double& t) const {
        Vec3 oc = ray.origin - center;
        double a = dot(ray.direction, ray.direction);
        double b = 2.0 * dot(oc, ray.direction);
        double c = dot(oc, oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return false;
        t = (-b - sqrt(discriminant)) / (2.0 * a);
        return t > 0;
    }
};

Vec3 color(const Ray& ray, const std::vector<Sphere>& spheres) {
    double closest_t = std::numeric_limits<double>::infinity();
    const Sphere* hit_sphere = nullptr;

    for (const auto& sphere : spheres) {
        double t;
        if (sphere.intersect(ray, t) && t < closest_t) {
            closest_t = t;
            hit_sphere = &sphere;
        }
    }

    if (hit_sphere) {
        Vec3 hit_point = ray.origin + ray.direction * closest_t;
        Vec3 normal = (hit_point - hit_sphere->center).normalize();
        Vec3 light_dir = Vec3(1, 1, -1).normalize();
        double diffuse = std::max(0.0, dot(normal, light_dir));
        
        // Simple shadow check
        Ray shadow_ray(hit_point + normal * 0.001, light_dir);
        for (const auto& sphere : spheres) {
            double t;
            if (sphere.intersect(shadow_ray, t)) {
                diffuse *= 0.5; // Soften shadows
                break;
            }
        }
        
        return hit_sphere->color * (diffuse * 0.7 + 0.3); // Add some ambient light
    }

    Vec3 unit_direction = ray.direction.normalize();
    double t = 0.5 * (unit_direction.y + 1.0);
    return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
}

int main() {
    int width = 800;
    int height = 400;
    int samples = 4;

    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3(0, 0, -1), 0.5, Vec3(0.7, 0.3, 0.3)));     // Red sphere
    spheres.push_back(Sphere(Vec3(0, -100.5, -1), 100, Vec3(0.3, 0.7, 0.3))); // Green "ground" sphere

    std::cout << "P3\n" << width << " " << height << "\n255\n";

    Vec3 lower_left_corner(-2, -1, -1);
    Vec3 horizontal(4, 0, 0);
    Vec3 vertical(0, 2, 0);
    Vec3 origin(0, 0, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            Vec3 col(0, 0, 0);
            for (int s = 0; s < samples; s++) {
                double u = double(i + dis(gen)) / double(width);
                double v = double(j + dis(gen)) / double(height);
                Ray r(origin, lower_left_corner + horizontal * u + vertical * v);
                col = col + color(r, spheres);
            }
            col = col * (1.0 / double(samples));
            
            int ir = int(255.99 * col.x);
            int ig = int(255.99 * col.y);
            int ib = int(255.99 * col.z);

            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    return 0;
}