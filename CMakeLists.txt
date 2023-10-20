cmake_minimum_required(VERSION 3.28)

project(01.unet)

add_executable(unet 
    main.cpp
    Net.cpp
    Net.h
    Layer.cpp
    Layer.h
)

#set(HDF5_ROOT "C:/Users/Evgen/Desktop/CMake-hdf5-1.14.2/CMake-hdf5-1.14.2/hdf5-1.14.2" CACHE PATH "C:/Users/Evgen/Desktop/CMake-hdf5-1.14.2/CMake-hdf5-1.14.2/hdf5-1.14.2")

# Find HDF5
#find_package(HDF5 REQUIRED COMPONENTS CXX)

# Add the HDF5 include directory to the include path
#include_directories(${HDF5_INCLUDE_DIRS})

# Link the HDF5 libraries to your executable
#target_link_libraries(main ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

set_target_properties(unet PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)