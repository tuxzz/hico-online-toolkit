g++ -std=c++17 -fopenmp -march=native -Ofast -funsafe-loop-optimizations -funsafe-math-optimizations -fassociative-math -freciprocal-math -ftree-vectorize -ffast-math -pipe -fno-stack-protector -flto -shared -Wl,-O1,--as-needed,-flto ediff.cpp -DNDEBUG -D_NDEBUG -DRELEASE -D_RELEASE -o fast_ediff.dll