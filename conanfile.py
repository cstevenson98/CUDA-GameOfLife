from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps

class CudaGolRecipe(ConanFile):
    name = "cuda-gol"
    version = "0.1"
    package_type = "application"

    # Optional metadata
    license = "MIT"
    author = "Conor"
    description = "CUDA-accelerated Game of Life simulation with OpenGL visualization"
    topics = ("cuda", "game-of-life", "opengl", "simulation")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe
    exports_sources = "CMakeLists.txt", "src/*", "imgui/*"

    def requirements(self):
        self.requires("glew/2.2.0")
        self.requires("glfw/3.3.8")  # This version provides the glfw3::glfw target
        self.requires("opengl/system")
        # ImGui is included in the project directly, so we don't need it from Conan

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        # Set CUDA compiler and architecture
        tc.variables["CMAKE_CUDA_COMPILER"] = "/usr/local/cuda/bin/nvcc"
        tc.variables["CMAKE_CUDA_ARCHITECTURES"] = "75"
        # Ensure GLFW provides the correct target name
        tc.variables["GLFW_BUILD_DOCS"] = "OFF"
        tc.variables["GLFW_BUILD_EXAMPLES"] = "OFF"
        tc.variables["GLFW_BUILD_TESTS"] = "OFF"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install() 