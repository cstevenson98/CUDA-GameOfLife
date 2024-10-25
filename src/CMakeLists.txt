# Make a library

add_library(${PROJECT_NAME} SHARED
    golCUDA.cu
    golPipeline.cu
    openGLutils.cpp
)

# Link the library to the executable

