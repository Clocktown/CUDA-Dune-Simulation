# CUDA-Dune-Simulation
This repository contains the full source code of our paper **Real-Time Desertscapes Simulation with CUDA**

# Setup
We provided a visual studio solution file. We have only tested our code on Windows, but it should be possible to get it to work on Linux. Mac might be an issue, as we are using OpenGL 4.6 for visualization.
## Requirements
The project is setup using CUDA 12.1 and `compute_75,sm75` flags. Older architectures and CUDA versions will work too, but you'll have to manually fix the visual studio solution to make it work.

We use a number of dependencies and recommend installation via VCPKG. The list of required packages for vcpkg is as follows:
`assimp,entt,stb,glm,glfw3,glad,imgui[glfw-binding,opengl3-binding],tinyexr,tinyfiledialogs,nlohmann-json`

# Code
The relevant simulation code is in `projects/dunes`. We further provide the scenes used in our paper in the `scenes` folder, which can be loaded with the application UI.