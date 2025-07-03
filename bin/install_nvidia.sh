
# clean system of any previously installed drivers
sudo apt-get --purge remove "nvidia-*"
sudo apt autoremove --purge
sudo apt clean

# Check for the right driver version and install
sudo apt install nvidia-driver-570-server-open


