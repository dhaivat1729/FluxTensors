# CMakeLists.txt

# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/linear_layer.cu
    src/relu.cu
)

# Create the executable
add_executable(my_executable ${SOURCES})

# Link any necessary libraries (if applicable)
# target_link_libraries(my_executable <library_name>)

# Instructions to compile and run the code:
# 1. Create a build directory:
#    mkdir build
#    cd build
# 2. Run CMake to configure the project:
#    cmake ..
# 3. Compile the project:
#    make
# 4. Run the executable:
#    ./my_executable
