cmake_minimum_required(VERSION 3.5)

project(01.main)

add_executable(main 
    main.cpp
)

set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)