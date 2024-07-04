#include "util.h"

__global__ void render_kernel(Triangle* transformed_triangles, int* triangle_offsets, int* triangle_counts,
                              unsigned char* textures, int* tex_widths, int* tex_heights,
                              unsigned char* output, float* zbuffer,
                              int width, int height, int num_objects, int num_scenes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int scene = blockIdx.z;
    if (x >= width || y >= height || scene >= num_scenes) return;

    int idx = (scene * height + y) * width + x;
    zbuffer[idx] = FLT_MAX;
    Vec3 color(0.2f, 0.2f, 0.2f);
    Vec3 light_dir(1, 1, 1);
    light_dir = light_dir.normalize();

    for (int obj = 0; obj < num_objects; obj++) {
        int triangle_offset = triangle_offsets[scene * num_objects + obj];
        for (int i = 0; i < triangle_counts[scene * num_objects + obj]; i++) {
            Triangle& tri = transformed_triangles[triangle_offset + i];
            
            Vec3 screen_coords[3];
            for (int j = 0; j < 3; j++) {
                screen_coords[j] = Vec3((tri.v[j].x + 1.0f) * width / 2.0f,
                                        (1.0f - tri.v[j].y) * height / 2.0f,
                                        tri.v[j].z);
            }

            Vec3 edge1 = screen_coords[1] - screen_coords[0];
            Vec3 edge2 = screen_coords[2] - screen_coords[0];
            Vec3 h = Vec3(x, y, 0) - screen_coords[0];
            float det = edge1.x * edge2.y - edge1.y * edge2.x;
            if (fabs(det) < 1e-6) continue;

            float u = (h.x * edge2.y - h.y * edge2.x) / det;
            float v = (edge1.x * h.y - edge1.y * h.x) / det;
            if (u < 0 || v < 0 || u + v > 1) continue;

            float z = screen_coords[0].z + u * (screen_coords[1].z - screen_coords[0].z) +
                      v * (screen_coords[2].z - screen_coords[0].z);
            if (z < zbuffer[idx]) {
                zbuffer[idx] = z;

                Vec2 uv = tri.uv[0] * (1-u-v) + tri.uv[1] * u + tri.uv[2] * v;
                int tex_x = uv.u * tex_widths[scene * num_objects + obj];
                int tex_y = (1.0f - uv.v) * tex_heights[scene * num_objects + obj];
                int tex_idx = (tex_y * tex_widths[scene * num_objects + obj] + tex_x) * 3;
                Vec3 tex_color(textures[tex_idx] / 255.0f,
                               textures[tex_idx + 1] / 255.0f,
                               textures[tex_idx + 2] / 255.0f);

                Vec3 normal = (tri.n[0] * (1-u-v) + tri.n[1] * u + tri.n[2] * v).normalize();
                float diffuse = max(0.0f, normal.dot(light_dir));
                
                color = tex_color * (0.3f + 0.7f * diffuse);
            }
        }
        textures += tex_widths[scene * num_objects + obj] * tex_heights[scene * num_objects + obj] * 3;
    }

    output[idx * 3 + 0] = static_cast<unsigned char>(min(color.x * 255.0f, 255.0f));
    output[idx * 3 + 1] = static_cast<unsigned char>(min(color.y * 255.0f, 255.0f));
    output[idx * 3 + 2] = static_cast<unsigned char>(min(color.z * 255.0f, 255.0f));
}

__global__ void transform_vertices_kernel(Triangle* input_triangles, Triangle* output_triangles, 
                                          int* triangle_offsets, int* triangle_counts, 
                                          Mat4* model_matrices, Mat4 projection, int num_objects, int num_scenes) {
    int scene = blockIdx.y;
    int obj = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scene >= num_scenes || obj >= num_objects) return;
    
    int offset = triangle_offsets[scene * num_objects + obj];
    int count = triangle_counts[scene * num_objects + obj];
    
    if (idx >= count) return;

    Triangle input_tri = input_triangles[offset + idx];
    Triangle& output_tri = output_triangles[offset + idx];
    Mat4 model = model_matrices[scene * num_objects + obj];
    Mat4 mp = projection * model;

    for (int j = 0; j < 3; j++) {
        output_tri.v[j] = mp.multiplyPoint(input_tri.v[j]);
        output_tri.n[j] = model.multiplyVector(input_tri.n[j]).normalize();
        output_tri.uv[j] = input_tri.uv[j];  // Copy UV coordinates
    }
}

int main() {
    const int width = 400, height = 300;
    const int num_objects = 2;
    const int num_scenes = 4;
    const int num_frames = 1000;
    const int fps = 60;
    
    std::vector<std::vector<Triangle>> triangles(num_objects);
    std::vector<unsigned char*> textures(num_objects);
    std::vector<int> tex_widths(num_objects), tex_heights(num_objects);

    // Load objects and textures
    load_obj("african_head.obj", triangles[0]);
    load_obj("drone.obj", triangles[1]);
    
    textures[0] = stbi_load("african_head_diffuse.tga", &tex_widths[0], &tex_heights[0], nullptr, 3);
    textures[1] = stbi_load("drone.png", &tex_widths[1], &tex_heights[1], nullptr, 3);
    
    // Prepare projection matrix
    Mat4 projection = create_projection_matrix(3.14159f / 4.0f, (float)width / height, 0.1f, 100.0f);

    // Define initial model matrices for all scenes
    Mat4 model_matrices[num_scenes][num_objects];
    for (int scene = 0; scene < num_scenes; scene++) {
        for (int obj = 0; obj < num_objects; obj++) {
            model_matrices[scene][obj] = create_model_matrix_random();
        }
    }

    std::vector<Triangle> all_triangles;
    std::vector<int> triangle_offsets(num_scenes * num_objects), triangle_counts(num_scenes * num_objects);
    std::vector<unsigned char> all_textures;
    std::vector<int> all_tex_widths(num_scenes * num_objects), all_tex_heights(num_scenes * num_objects);

    for (int scene = 0; scene < num_scenes; scene++) {
        for (int i = 0; i < num_objects; i++) {
            triangle_offsets[scene * num_objects + i] = all_triangles.size();
            triangle_counts[scene * num_objects + i] = triangles[i].size();
            
            all_triangles.insert(all_triangles.end(), triangles[i].begin(), triangles[i].end());
            
            all_tex_widths[scene * num_objects + i] = tex_widths[i];
            all_tex_heights[scene * num_objects + i] = tex_heights[i];
            all_textures.insert(all_textures.end(), textures[i], textures[i] + tex_widths[i] * tex_heights[i] * 3);
        }
    }

    // Allocate GPU memory
    Triangle *d_input_triangles, *d_transformed_triangles;
    int *d_triangle_offsets, *d_triangle_counts, *d_tex_widths, *d_tex_heights;
    unsigned char *d_textures, *d_output;
    float* d_zbuffer;
    Mat4* d_model_matrices;

    CHECK_CUDA(cudaMalloc(&d_input_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_transformed_triangles, all_triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_triangle_offsets, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_triangle_counts, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_textures, all_textures.size() * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_tex_widths, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tex_heights, num_scenes * num_objects * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, num_scenes * width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_zbuffer, num_scenes * width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_model_matrices, num_scenes * num_objects * sizeof(Mat4)));

    // Copy static data to GPU
    CHECK_CUDA(cudaMemcpy(d_input_triangles, all_triangles.data(), all_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_offsets, triangle_offsets.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_triangle_counts, triangle_counts.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_textures, all_textures.data(), all_textures.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_widths, all_tex_widths.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tex_heights, all_tex_heights.data(), num_scenes * num_objects * sizeof(int), cudaMemcpyHostToDevice));

    // Set up kernel configurations
    int max_triangles = *std::max_element(triangle_counts.begin(), triangle_counts.end());
    dim3 transform_block_size(256);
    dim3 transform_grid_size((max_triangles + transform_block_size.x - 1) / transform_block_size.x, num_scenes, num_objects);
    
    dim3 render_block_size(16, 16, 1);
    dim3 render_grid_size((width + render_block_size.x - 1) / render_block_size.x, 
                          (height + render_block_size.y - 1) / render_block_size.y, 
                          num_scenes);

    // Prepare video writers for each scene
    std::vector<cv::VideoWriter> video_writers(num_scenes);
    for (int scene = 0; scene < num_scenes; scene++) {
        std::string filename = "output_scene" + std::to_string(scene) + ".mp4";
        video_writers[scene].open(filename, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(width, height));
        if (!video_writers[scene].isOpened()) {
            std::cerr << "Could not open the output video file for writing: " << filename << std::endl;
            return -1;
        }
    }

    // Main rendering loop
    std::vector<unsigned char> output(num_scenes * width * height * 3);
    for (int frame = 0; frame < num_frames; frame++) {
        // Update model matrices (rotate objects)
        for (int scene = 0; scene < num_scenes; scene++) {
            for (int obj = 0; obj < num_objects; obj++) {
                float rotation = 0.1f;  // Adjust rotation speed as needed
                
                // Create rotation matrix
                Mat4 rotation_matrix(
                    cos(rotation), 0, -sin(rotation), 0,
                    0, 1, 0, 0,
                    sin(rotation), 0, cos(rotation), 0,
                    0, 0, 0, 1
                );

                // Extract the current translation
                Vec3 translation(model_matrices[scene][obj].m[3], 
                                 model_matrices[scene][obj].m[7], 
                                 model_matrices[scene][obj].m[11]);

                // Apply rotation to the orientation part of the model matrix
                model_matrices[scene][obj] = rotation_matrix * model_matrices[scene][obj];

                // Restore the original translation
                model_matrices[scene][obj].m[3] = translation.x;
                model_matrices[scene][obj].m[7] = translation.y;
                model_matrices[scene][obj].m[11] = translation.z;
            }
        }

        // Copy updated model matrices to GPU
        CHECK_CUDA(cudaMemcpy(d_model_matrices, model_matrices, num_scenes * num_objects * sizeof(Mat4), cudaMemcpyHostToDevice));

        // Transform vertices
        transform_vertices_kernel<<<transform_grid_size, transform_block_size>>>(
            d_input_triangles, d_transformed_triangles, d_triangle_offsets, d_triangle_counts,
            d_model_matrices, projection, num_objects, num_scenes);

        // Render scenes
        render_kernel<<<render_grid_size, render_block_size>>>(
            d_transformed_triangles, d_triangle_offsets, d_triangle_counts,
            d_textures, d_tex_widths, d_tex_heights,
            d_output, d_zbuffer, width, height, num_objects, num_scenes);

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(output.data(), d_output, num_scenes * width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        // Write frames for each scene
        for (int scene = 0; scene < num_scenes; scene++) {
            cv::Mat frame(height, width, CV_8UC3, output.data() + scene * width * height * 3);
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);  // OpenCV uses BGR by default
            video_writers[scene].write(frame);
        }
    }

    // Close video writers
    for (auto& writer : video_writers) {
        writer.release();
    }

    // Clean up GPU memory
    cudaFree(d_input_triangles);
    cudaFree(d_transformed_triangles);
    cudaFree(d_triangle_offsets);
    cudaFree(d_triangle_counts);
    cudaFree(d_textures);
    cudaFree(d_tex_widths);
    cudaFree(d_tex_heights);
    cudaFree(d_output);
    cudaFree(d_zbuffer);
    cudaFree(d_model_matrices);

    // Clean up textures
    for (auto texture : textures) {
        stbi_image_free(texture);
    }

    return 0;
}