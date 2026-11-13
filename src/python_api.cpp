// src/python_api.cpp
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#include "pointpillar.hpp"
#include "common/check.hpp"

// Global state to hold the engine and CUDA stream in memory between Python calls
std::shared_ptr<pointpillar::lidar::Core> global_core = nullptr;
cudaStream_t global_stream;

extern "C" {
    
    // 1. Initialize the engine once when Python starts
    void init_engine(const char* plan_file_path) {
        std::cout << "Loading TensorRT Engine from: " << plan_file_path << std::endl;

        // Voxelization parameters explicitly pulled from the original main.cpp
        pointpillar::lidar::VoxelizationParameter vp;
        vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);
        vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);
        vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
        vp.grid_size = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
        vp.max_voxels = 40000;
        vp.max_points_per_voxel = 32;
        vp.max_points = 300000;
        vp.num_feature = 4;

        // Post-processing parameters
        pointpillar::lidar::PostProcessParameter pp;
        pp.min_range = vp.min_range;
        pp.max_range = vp.max_range;
        pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);

        // Core compilation
        pointpillar::lidar::CoreParameter param;
        param.voxelization = vp;
        param.lidar_model = plan_file_path; // Replaced hardcoded path with Python argument
        param.lidar_post = pp;
        
        global_core = pointpillar::lidar::create_core(param);
        
        if (global_core == nullptr) {
            std::cerr << "Core initialization failed!" << std::endl;
            return;
        }

        // Initialize the persistent CUDA execution stream
        cudaStreamCreate(&global_stream);
        global_core->print();
    }

    // 2. Run inference on live UDP frames from Python
    int run_inference(float* points_array, int num_points, float* out_boxes) {
        if (global_core == nullptr) {
            std::cerr << "Engine not initialized. Call init_engine() first." << std::endl;
            return 0;
        }

        // Pass the raw memory pointer directly to the GPU
        auto bboxes = global_core->forward(points_array, num_points, global_stream);
        
        // Wait for the GPU to finish execution before reading the results
        checkRuntime(cudaStreamSynchronize(global_stream)); 

        // Flatten the bounding box structs into the contiguous out_boxes array
        int num_detected = bboxes.size();
        for(int i = 0; i < num_detected; i++) {
            out_boxes[i*9 + 0] = bboxes[i].x;
            out_boxes[i*9 + 1] = bboxes[i].y;
            out_boxes[i*9 + 2] = bboxes[i].z;
            out_boxes[i*9 + 3] = bboxes[i].w;
            out_boxes[i*9 + 4] = bboxes[i].l;
            out_boxes[i*9 + 5] = bboxes[i].h;
            out_boxes[i*9 + 6] = bboxes[i].rt;
            out_boxes[i*9 + 7] = bboxes[i].id;
            out_boxes[i*9 + 8] = bboxes[i].score;
        }
        
        return num_detected;
    }
}