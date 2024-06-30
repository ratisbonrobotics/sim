#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vertex {
    glm::vec3 position;
    glm::vec2 texCoords;
    glm::vec3 normal;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    GLuint VAO, VBO, EBO;
    GLuint texture;
};

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoords;
    layout (location = 2) in vec3 aNormal;
    
    out vec2 TexCoords;
    out vec3 Normal;
    out vec3 FragPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;  
        TexCoords = aTexCoords;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    in vec2 TexCoords;
    in vec3 Normal;
    in vec3 FragPos;
    
    uniform sampler2D texture_diffuse1;
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    
    void main()
    {    
        vec3 color = texture(texture_diffuse1, TexCoords).rgb;
        
        // Ambient
        vec3 ambient = 0.2 * color;
        
        // Diffuse 
        vec3 lightDir = normalize(lightPos - FragPos);
        vec3 normal = normalize(Normal);
        float diff = max(dot(lightDir, normal), 0.0);
        vec3 diffuse = diff * color;
        
        // Specular
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = vec3(0.2) * spec;
        
        FragColor = vec4(ambient + diffuse + specular, 1.0);
    }
)";

void loadOBJ(const char* filename, Mesh& mesh) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filename << std::endl;
        return;
    }

    std::vector<glm::vec3> temp_vertices, temp_normals;
    std::vector<glm::vec2> temp_texcoords;
    std::string line, type;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> type;

        if (type == "v") {
            glm::vec3 v;
            iss >> v.x >> v.y >> v.z;
            temp_vertices.push_back(v);
        } else if (type == "vt") {
            glm::vec2 vt;
            iss >> vt.x >> vt.y;
            temp_texcoords.push_back(vt);
        } else if (type == "vn") {
            glm::vec3 vn;
            iss >> vn.x >> vn.y >> vn.z;
            temp_normals.push_back(vn);
        } else if (type == "f") {
            for (int i = 0; i < 3; i++) {
                Vertex vertex;
                unsigned int v, vt, vn;
                char slash;
                iss >> v >> slash >> vt >> slash >> vn;
                vertex.position = temp_vertices[v - 1];
                vertex.texCoords = temp_texcoords[vt - 1];
                vertex.normal = temp_normals[vn - 1];
                mesh.vertices.push_back(vertex);
                mesh.indices.push_back(mesh.indices.size());
            }
        }
    }
}

void setupMesh(Mesh& mesh) {
    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);
  
    glBindVertexArray(mesh.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(Vertex), &mesh.vertices[0], GL_STATIC_DRAW);  

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW);

    // vertex positions
    glEnableVertexAttribArray(0);	
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // vertex texture coords
    glEnableVertexAttribArray(1);	
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    // vertex normals
    glEnableVertexAttribArray(2);	
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

    glBindVertexArray(0);
}

void loadTexture(const char* filename, GLuint& texture) {
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    unsigned char *data = stbi_load(filename, &width, &height, &nrChannels, 0);
    if (data) {
        GLenum format = nrChannels == 3 ? GL_RGB : GL_RGBA;
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cerr << "Failed to load texture: " << filename << std::endl;
    }
    stbi_image_free(data);
}

GLuint createShaderProgram() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

int main() {
    const int width = 800, height = 600;

    // EGL setup
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display, NULL, NULL);

    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    EGLConfig config;
    EGLint numConfigs;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE,
    };

    surface = eglCreatePbufferSurface(display, config, pbufferAttribs);

    eglBindAPI(EGL_OPENGL_API);


    const char* eglExtensions = eglQueryString(display, EGL_EXTENSIONS);
std::cout << "EGL Extensions: " << eglExtensions << std::endl;

EGLint contextAttribs[] = {
    EGL_CONTEXT_MAJOR_VERSION, 3,
    EGL_CONTEXT_MINOR_VERSION, 3,
    EGL_NONE
};

context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
if (context == EGL_NO_CONTEXT) {
    std::cerr << "Failed to create EGL context. Error: " << eglGetError() << std::endl;
    return -1;
}

if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
    std::cerr << "Failed to make context current. Error: " << eglGetError() << std::endl;
    return -1;
}

std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    std::cout << "GLEW Version: " << glewGetString(GLEW_VERSION) << std::endl;

    // Create framebuffer
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Create texture to render to
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    // Create renderbuffer for depth and stencil
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is not complete!" << std::endl;
        return -1;
    }

    // Load meshes and textures
    Mesh africanHead, drone;
    loadOBJ("african_head.obj", africanHead);
    loadOBJ("drone.obj", drone);
    setupMesh(africanHead);
    setupMesh(drone);
    loadTexture("african_head_diffuse.tga", africanHead.texture);
    loadTexture("drone.png", drone.texture);

    // Create and use shader program
    GLuint shaderProgram = createShaderProgram();
    glUseProgram(shaderProgram);

    // Set up view and projection matrices
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Set up lighting
    glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 1.0f, 1.0f, 2.0f);
    glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), 0.0f, 0.0f, 3.0f);

    // Render
    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render African Head
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-1.0f, 0.0f, -3.0f));
    model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glBindVertexArray(africanHead.VAO);
    glBindTexture(GL_TEXTURE_2D, africanHead.texture);
    glDrawElements(GL_TRIANGLES, africanHead.indices.size(), GL_UNSIGNED_INT, 0);

    // Render Drone (continued)
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.0f, 0.5f, -2.5f));
    model = glm::scale(model, glm::vec3(0.1f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glBindVertexArray(drone.VAO);
    glBindTexture(GL_TEXTURE_2D, drone.texture);
    glDrawElements(GL_TRIANGLES, drone.indices.size(), GL_UNSIGNED_INT, 0);

    // Read pixels and save to file
    unsigned char* pixels = new unsigned char[3 * width * height];
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    // Flip the image vertically (OpenGL has (0,0) at bottom left, image formats typically have it at top left)
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width * 3; x++) {
            std::swap(pixels[y * width * 3 + x], pixels[(height - 1 - y) * width * 3 + x]);
        }
    }

    stbi_write_png("output.png", width, height, 3, pixels, width * 3);

    delete[] pixels;

    // Clean up OpenGL resources
    glDeleteVertexArrays(1, &africanHead.VAO);
    glDeleteBuffers(1, &africanHead.VBO);
    glDeleteBuffers(1, &africanHead.EBO);
    glDeleteTextures(1, &africanHead.texture);

    glDeleteVertexArrays(1, &drone.VAO);
    glDeleteBuffers(1, &drone.VBO);
    glDeleteBuffers(1, &drone.EBO);
    glDeleteTextures(1, &drone.texture);

    glDeleteProgram(shaderProgram);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &texture);
    glDeleteRenderbuffers(1, &rbo);

    // EGL cleanup
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
    eglDestroySurface(display, surface);
    eglTerminate(display);

    return 0;
}