# /etc/sysctl.conf

# --- Huge Pages ---
# Reserve 112GB of RAM as static 2MB Huge Pages.
# (112 * 1024) / 2 = 57344
vm.nr_hugepages = 57344
vm.hugetlb_shm_group = 1000 # Your user's group ID

# --- Memory Management ---
# Avoid swapping at all costs.
vm.swappiness = 1
# Prevent cross-NUMA node memory reclamation.
vm.zone_reclaim_mode = 0
# Allow memory overcommitment for large model allocations.
vm.overcommit_memory = 1

# --- Cache and I/O Tuning ---
# Prioritize application memory over filesystem cache.
vm.vfs_cache_pressure = 50
# Write back dirty pages quickly to avoid I/O stalls.
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
# Proactively compact memory to help allocation success.
vm.compact_memory = 1



# Memory Allocator 
# VERY IMPORTANT: Need to run the export command after doing docker build; find Out why 
# Install jemalloc
sudo apt install libjemalloc-dev

# Set as default allocator
# this seems to be the fastest one
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Or for tcmalloc:
sudo apt install libtcmalloc-minimal4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4


# Disable THP (can cause fragmentation)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag




# Install CPU frequency utilities
sudo apt install cpufrequtils

# Set performance governor for all cores
sudo cpupower frequency-set -g performance

# Make persistent across reboots
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils





GRUB_CMDLINE_LINUX="isolcpus=4-15,20-31 nohz_full=4-15,20-31 rcu_nocbs=4-15,20-31 idle=poll rcu_nocb_poll transparent_hugepage=never audit=0 nmi_watchdog=0"

sudo update-grub
sudo reboot
