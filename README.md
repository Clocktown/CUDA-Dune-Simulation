# CUDA-Dune-Simulation
This repository contains the full source code of our *GRAPP 2024* paper **Real-Time Desertscapes Simulation with CUDA** found [here](https://www.scitepress.org/PublicationsDetail.aspx?ID=dywC2RBocrQ=&t=1).
The source code has since been extended for submission of an extended paper for *Communications in Computer and Information Science (CCIS)*, titled **Real-Time Desertscapes Simulation with Reptation and Divergence-Free Wind Fields**. For this extended paper, we added divergence-free wind fields via a FFT-based method and improved the previously artifact-ridden approach to reptation with a new method. The expected publication date is January 07, 2025.



https://github.com/user-attachments/assets/141866a9-8640-4b41-85de-5ad28fbe988d


https://github.com/user-attachments/assets/1970eaca-e13e-4c2a-847a-2966dc29448f


https://github.com/user-attachments/assets/519302c1-d33d-4519-bfd7-c3db25601e4d


https://github.com/user-attachments/assets/86296054-10af-405a-84fd-8bef145dffd5


https://github.com/user-attachments/assets/94312da8-92e8-4649-b1fa-c22325162e7c







# Setup
We provided a visual studio solution file. We have only tested our code on Windows, but it should be possible to get it to work on Linux. Mac might be an issue, as we are using OpenGL 4.6 for visualization.
## Requirements
The project is setup using Visual Studio 2022, CUDA 12.5 and `compute_75,sm75` flags. Older architectures and CUDA versions will work too, but you'll have to manually fix the visual studio solution to make it work.

If you are running the executable on a Laptop, make sure that it runs on your dedicated NVIDIA graphics card instead of i.e. the Intel integrated GPU. While not a problem for CUDA, the OpenGL part of our application will likely not work properly with these GPUs. Additionally,
we are sharing OpenGL textures which CUDA, which could cause additional problems if OpenGL runs on a different GPU compared to CUDA kernels.

A `vcpkg.json` manifest file has been provided. Default installs of Visual Studio 2022 already include VCPKG and should install the required packages automatically.

The list of required packages for vcpkg is as follows (in case that manual installation is needed/required):
`assimp,entt,stb,glm,glfw3,glad,imgui[glfw-binding,opengl3-binding],tinyexr,tinyfiledialogs,nlohmann-json`

# Code
The relevant simulation code is in `projects/dunes`. We further provide the scenes used in our paper in the `scenes` folder, which can be loaded with the application UI.
