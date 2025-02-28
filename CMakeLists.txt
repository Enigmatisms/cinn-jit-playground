cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

set(PROJ_NAME "isolated")
project(${PROJ_NAME} LANGUAGES CUDA CXX)

set(CUDA_COMMON_FLAGS --expt-relaxed-constexpr --extended-lambda)

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	set(CMAKE_BUILD_TYPE Debug)
	if (MSVC)
		message(STATUS "MSVC: Building project in DEBUG mode.")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -g -G")
		set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -lineinfo")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /g /Od /Zi /MDd")
	else()
		message(STATUS "GCC & G++: Building project in DEBUG mode.")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -G -O0 -lineinfo")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
	endif ()
    set(CUDA_CONFIG_FLAGS -G)
ELSE()
	# to check register usage, please set "-Xptxas -v"
	# to cap the register usage to boost occupancy, set ""
	set(CMAKE_BUILD_TYPE Release)
	if (MSVC)
		message(STATUS "MSVC: Building project in RELEASE mode.")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -arch=sm_70")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
	else()
		message(STATUS "GCC & G++: Building project in RELEASE mode.")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -arch=sm_70")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif ()
ENDIF()
set(CUDA_CONFIG_FLAGS -arch=sm_70 --fmad=true -std=c++14)
set(CUDA_FINAL_FLAGS ${CUDA_COMMON_FLAGS} ${CUDA_CONFIG_FLAGS})
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
)

message(STATUS ${CMAKE_CURRENT_LIST_DIR})

set(EXEC1_NAME "manual")
add_executable(${EXEC1_NAME}
	tests/trial_manual.cu
)

target_compile_options(
	${EXEC1_NAME} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FINAL_FLAGS}>
)

aux_source_directory(./src SRC)
set(EXEC2_NAME "trial_batched")
add_executable(${EXEC2_NAME}
	tests/trial_batched.cu
	${SRC}
)

target_compile_options(
	${EXEC2_NAME} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FINAL_FLAGS}>
)