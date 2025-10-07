# EXPERIMENTAL: 1GB Huge Pages for Large Model Inference

**STATUS: EXPERIMENTAL - Advanced Memory Optimization**

**WARNING**: 1GB huge pages require boot-time configuration and may impact system stability. This optimization is most beneficial for models larger than 15GB.

## Table of Contents

- [EXPERIMENTAL: 1GB Huge Pages for Large Model Inference](#experimental-1gb-huge-pages-for-large-model-inference)
  - [Table of Contents](#table-of-contents)
  - [Executive Summary](#executive-summary)
  - [Technical Background](#technical-background)
    - [TLB Hierarchy on Zen 5](#tlb-hierarchy-on-zen-5)
    - [Memory Bandwidth Impact](#memory-bandwidth-impact)
  - [Implementation Design](#implementation-design)
    - [Kernel Configuration](#kernel-configuration)
      - [Boot-Time Setup](#boot-time-setup)
      - [Runtime Allocation (Less Reliable)](#runtime-allocation-less-reliable)
    - [Wrapper Modifications](#wrapper-modifications)
      - [Updated hugepage_mmap_wrapper.cpp](#updated-hugepage_mmap_wrappercpp)
    - [Mount Configuration](#mount-configuration)
      - [Dual Mount Strategy](#dual-mount-strategy)
    - [Management Script Updates](#management-script-updates)
      - [Modified manage-hugepages-models.sh](#modified-manage-hugepages-modelssh)
  - [Performance Analysis](#performance-analysis)
    - [TLB Miss Reduction](#tlb-miss-reduction)
    - [Memory Access Patterns](#memory-access-patterns)
  - [Integration Testing](#integration-testing)
    - [Verification Commands](#verification-commands)
    - [Performance Validation](#performance-validation)
  - [Risk Mitigation](#risk-mitigation)
    - [Memory Fragmentation](#memory-fragmentation)
    - [System Impact](#system-impact)
  - [Compatibility Matrix](#compatibility-matrix)
  - [Alternative Approaches](#alternative-approaches)
    - [Transparent Huge Pages (THP)](#transparent-huge-pages-thp)
    - [Reserved Memory Regions](#reserved-memory-regions)
    - [NUMA Balancing](#numa-balancing)
  - [Success Metrics](#success-metrics)
  - [Implementation Timeline](#implementation-timeline)
    - [Day 1: Kernel Configuration](#day-1-kernel-configuration)
    - [Day 2: Wrapper Modification](#day-2-wrapper-modification)
    - [Day 3: Testing and Validation](#day-3-testing-and-validation)
  - [Conclusion](#conclusion)

## Executive Summary

Extension of the existing 2MB huge pages implementation to support 1GB pages, reducing TLB pressure by 512x compared to 2MB pages. For a 30GB model, only 30 TLB entries are needed instead of 15,360.

**Expected Impact**: 5-10% additional inference speedup over 2MB pages
**Engineering Effort**: 2-3 days (builds on existing wrapper)
**Risk Level**: Medium
**Compatibility**: Requires kernel 3.8 or newer and processor support

## Technical Background

### TLB Hierarchy on Zen 5

```
Standard 4KB pages:  7,864,320 pages for 30GB model
2MB huge pages:      15,360 pages for 30GB model  
1GB huge pages:      30 pages for 30GB model

Zen 5 TLB Structure:
- L1 DTLB: 72 entries (4KB), 64 entries (2MB), 8 entries (1GB)
- L2 TLB: 3072 entries (4KB), 2048 entries (2MB/1GB shared)
```

With 1GB pages, the entire 30GB model fits in L1 DTLB (8 entries) with room to spare.

### Memory Bandwidth Impact

Fewer TLB misses means:
- Less time in page table walks
- More memory bandwidth for actual data
- Better CPU pipeline utilization

## Implementation Design

### Kernel Configuration

#### Boot-Time Setup

```bash
# /etc/default/grub modification
GRUB_CMDLINE_LINUX="hugepagesz=1G hugepages=32 default_hugepagesz=1G"

# Apply changes
sudo update-grub
sudo reboot

# Verify after reboot
grep Huge /proc/meminfo
# HugePages_Total:       32
# HugePages_Free:        32
# Hugepagesize:     1048576 kB  # 1GB
```

#### Runtime Allocation (Less Reliable)

```bash
# Try to allocate 1GB pages at runtime
echo 32 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages

# Often fails due to memory fragmentation
# Boot-time allocation strongly recommended
```

### Wrapper Modifications

#### Updated hugepage_mmap_wrapper.cpp

```cpp
// Add 1GB page support detection
static size_t get_hugepage_size(size_t file_size) {
    // Try 1GB pages for files >= 1GB
    if (file_size >= (1ULL << 30)) {
        // Check if 1GB pages are available
        FILE* fp = fopen("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages", "r");
        if (fp) {
            int free_1gb = 0;
            fscanf(fp, "%d", &free_1gb);
            fclose(fp);
            
            if (free_1gb > 0) {
                return (1ULL << 30);  // 1GB
            }
        }
    }
    
    // Fallback to 2MB pages
    return (1ULL << 21);  // 2MB
}

// Modified mmap interception
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    init_functions();
    
    if (fd >= 0 && is_hugetlbfs_fd(fd)) {
        struct stat st;
        if (fstat(fd, &st) == 0 && offset == 0 && length == (size_t)st.st_size) {
            
            size_t hugepage_size = get_hugepage_size(length);
            int map_flags = MAP_PRIVATE | MAP_ANONYMOUS;
            
            if (hugepage_size == (1ULL << 30)) {
                // Use MAP_HUGE_1GB for 1GB pages
                map_flags |= MAP_HUGETLB | MAP_HUGE_1GB;
                fprintf(stderr, "INFO: Using 1GB huge pages for %.2f GB model\n",
                        length / (1024.0 * 1024.0 * 1024.0));
            } else {
                // Use MAP_HUGE_2MB for 2MB pages
                map_flags |= MAP_HUGETLB | MAP_HUGE_2MB;
                fprintf(stderr, "INFO: Using 2MB huge pages for %.2f GB model\n",
                        length / (1024.0 * 1024.0 * 1024.0));
            }
            
            void* huge_mem = real_mmap(nullptr, length, 
                                      PROT_READ | PROT_WRITE,
                                      map_flags, -1, 0);
            
            if (huge_mem == MAP_FAILED) {
                // Fallback cascade: 1GB -> 2MB -> 4KB
                if (hugepage_size == (1ULL << 30)) {
                    fprintf(stderr, "WARNING: 1GB allocation failed, trying 2MB\n");
                    map_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB;
                    huge_mem = real_mmap(nullptr, length, 
                                       PROT_READ | PROT_WRITE,
                                       map_flags, -1, 0);
                }
                
                if (huge_mem == MAP_FAILED) {
                    fprintf(stderr, "WARNING: Huge page allocation failed, using regular pages\n");
                    huge_mem = real_mmap(nullptr, length,
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                }
            }
            
            // Rest of implementation remains the same...
        }
    }
    
    return real_mmap(addr, length, prot, flags, fd, offset);
}
```

### Mount Configuration

#### Dual Mount Strategy

```bash
# Create separate mount points for different page sizes
sudo mkdir -p /mnt/hugepages-2mb
sudo mkdir -p /mnt/hugepages-1gb

# Mount 2MB pages
sudo mount -t hugetlbfs -o pagesize=2M,size=10G none /mnt/hugepages-2mb

# Mount 1GB pages
sudo mount -t hugetlbfs -o pagesize=1G,size=32G none /mnt/hugepages-1gb

# Add to /etc/fstab
echo "none /mnt/hugepages-2mb hugetlbfs pagesize=2M,size=10G 0 0" | sudo tee -a /etc/fstab
echo "none /mnt/hugepages-1gb hugetlbfs pagesize=1G,size=32G 0 0" | sudo tee -a /etc/fstab
```

### Management Script Updates

#### Modified manage-hugepages-models.sh

```bash
# Add page size selection
load_model() {
    local model_path="$1"
    local page_size="${2:-auto}"  # auto, 2mb, or 1gb
    
    # Determine optimal page size
    local model_size=$(stat -c%s "${source_file}")
    local model_size_gb=$((model_size / 1024 / 1024 / 1024))
    
    if [[ "$page_size" == "auto" ]]; then
        if [[ $model_size_gb -ge 10 ]]; then
            page_size="1gb"
        else
            page_size="2mb"
        fi
    fi
    
    # Select mount point based on page size
    if [[ "$page_size" == "1gb" ]]; then
        HUGEPAGES_MOUNT="/mnt/hugepages-1gb"
        echo "Using 1GB huge pages for model"
    else
        HUGEPAGES_MOUNT="/mnt/hugepages-2mb"
        echo "Using 2MB huge pages for model"
    fi
    
    # Rest of loading logic...
}
```

## Performance Analysis

### TLB Miss Reduction

```
30GB Model TLB Misses (estimated):

4KB pages:  ~1,000,000 misses per inference
2MB pages:  ~2,000 misses per inference
1GB pages:  ~4 misses per inference

Theoretical speedup from TLB alone: 5-10%
```

### Memory Access Patterns

```cpp
// Benchmark to measure TLB impact
void benchmark_tlb_impact(void* model_data, size_t size) {
    // Random access pattern (worst case for TLB)
    for (int i = 0; i < 1000000; i++) {
        size_t offset = (rand() % (size / 64)) * 64;
        volatile char val = ((char*)model_data)[offset];
    }
    
    // Sequential access (best case)
    for (size_t i = 0; i < size; i += 64) {
        volatile char val = ((char*)model_data)[i];
    }
}
```

## Integration Testing

### Verification Commands

```bash
# Check 1GB page allocation
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages

# Monitor during model load
watch -n 1 'grep Huge /proc/meminfo'

# Check process usage
cat /proc/$(pgrep llama-server)/status | grep HugetlbPages
```

### Performance Validation

```bash
#!/bin/bash
# Compare page sizes

# Baseline - 2MB pages
echo "Testing with 2MB pages..."
sudo ./manage-hugepages-models.sh load model.gguf 2mb
time docker run --rm llama-cpu-0 benchmark

# Test - 1GB pages  
echo "Testing with 1GB pages..."
sudo ./manage-hugepages-models.sh load model.gguf 1gb
time docker run --rm llama-cpu-0 benchmark

# Measure:
# - Model load time
# - Inference tokens/second
# - Memory bandwidth (via perf)
```

## Risk Mitigation

### Memory Fragmentation

**Problem**: 1GB pages require contiguous 1GB memory regions

**Solutions**:
1. Allocate at boot time (most reliable)
2. Drop caches before allocation: `echo 3 > /proc/sys/vm/drop_caches`
3. Use memory compaction: `echo 1 > /proc/sys/vm/compact_memory`
4. Automatic fallback to 2MB pages

### System Impact

**Concerns**:
- 32GB permanently reserved for huge pages
- Other applications can't use this memory
- System might become memory constrained

**Mitigations**:
- Use dynamic allocation when possible
- Monitor system memory pressure
- Implement automatic page release

## Compatibility Matrix

| Component | 2MB Pages | 1GB Pages | Notes |
|-----------|-----------|-----------|-------|
| Kernel 3.8 or newer | Yes | Yes | Full support |
| Zen 5 CPU | Yes | Yes | 8 dedicated 1GB TLB entries |
| Docker | Yes | Yes | Transparent to containers |
| llama.cpp | Yes | Yes | Via wrapper |

## Alternative Approaches

### Transparent Huge Pages (THP)
- Pros: Automatic, no configuration
- Cons: Unpredictable, you disabled it

### Reserved Memory Regions
- Pros: Guaranteed contiguous memory
- Cons: Complex kernel configuration

### NUMA Balancing
- Pros: Automatic optimization
- Cons: Single NUMA node in your system

## Success Metrics

1. **Allocation Success**: 1GB pages allocated without fragmentation
2. **TLB Hit Rate**: Greater than 99.9% (measured via perf)
3. **Performance Gain**: 5-10% over 2MB pages
4. **Stability**: No OOM or allocation failures

## Implementation Timeline

### Day 1: Kernel Configuration
- Boot parameter setup
- Mount point configuration
- Verification of 1GB page support

### Day 2: Wrapper Modification
- Add MAP_HUGE_1GB support
- Implement fallback logic
- Update management scripts

### Day 3: Testing and Validation
- Performance benchmarking
- Stability testing
- Documentation updates

## Conclusion

1GB huge pages offer meaningful performance improvements for large models with minimal implementation effort. The main challenge is memory allocation at boot time rather than code complexity.

---

**Note**: This optimization requires careful system configuration. Test thoroughly before deployment.

*Last Updated: 2025-09-23*