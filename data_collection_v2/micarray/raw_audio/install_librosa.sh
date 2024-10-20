#wget -qO - https://ftp-master.debian.org/keys/archive-key-10.asc | sudo apt-key add -
#echo deb http://deb.debian.org/debian buster main contrib non-free | sudo tee -a /etc/apt/sources.list
#sudo apt-get update
#sudo apt-get install llvm-7 -y
echo $(which llvm-config-7)
#/home/vax/venv-vax/bin/python -m pip install soundfile sounddevice pyaudio
#LLVM_CONFIG=/usr/bin/llvm-config-7 /home/vax/venv-vax/bin/python -m pip install llvmlite==0.32
#LLVM_CONFIG=/usr/bin/llvm-config-7 /home/vax/venv-vax/bin/python -m pip install numba==0.49
#sudo apt install libatlas-base -y
LLVM_CONFIG=/usr/bin/llvm-config-7 /home/vax/venv-vax/bin/pip install -U llvmlite==0.31.0 numba==0.48.0 colorama==0.3.9 librosa==0.6.3
#LLVM_CONFIG=/usr/bin/llvm-config-7 /home/vax/venv-vax/bin/python -m pip install librosa==0.10
