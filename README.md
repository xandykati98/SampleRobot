# =============== Linux ===============

# MobileSim install:
sudo apt install ./mobilesim_0.9.8+ubuntu16_amd64.deb 

# C++:
sudo apt update && sudo apt install g++
C/C++ vscode extension (Microsoft)


# ARIA
sudo dpkg -i libaria_2.9.4+ubuntu16_amd64.deb 

# Build
make
chmod +x ./build/main

# Run
./build/main
  
# Delete the build
make clean

# =============== Neural Network Training ===============

# Python setup (for training):
cd python_training
pip install -r requirements.txt

# Train with Reinforcement Learning (RECOMMENDED):
py train_rl.py
py export_weights_rl.py

# OR Train with Evolutionary Algorithm:
py train.py
py export_weights.py

# Copy weights to project root:
cp weights.json ..

# Rebuild and run with neural network:
cd ..
make clean && make
./build/main

# =============== Windows ===============

Install on this order:

msys2-x86_64-20250830.exe

MobileSim-0.7.5.exe

ARIA-2.9.1-1-x64.exe


Open the MSYS terminal an run:

Installing the compiler:
pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-gcc

Testing the installation
export PATH="/c/msys64/ucrt64/bin:$PATH"
g++ --version

Adding to the PATH, permanently
nano ~/.bashrc
export PATH="/c/msys64/ucrt64/bin:$PATH"   //(Add this to the end of the file)
source ~/.bashrc
