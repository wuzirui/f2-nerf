rm build -rf
mkdir build
cmake . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cd build
cmake --build build --target main --config RelWithDebInfo -j
