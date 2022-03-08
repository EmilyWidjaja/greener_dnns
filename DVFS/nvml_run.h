
   
#ifndef GPU_NVML_RUN
#define GPU_NVML_RUN

void nvml_setup(int device, int core_clock, int mem_clock);
void nvml_reset();
int read_int(const char* path);

int main();

// unsigned int assign_clock(char *deviceName);

#endif