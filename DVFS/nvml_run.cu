#include "nvml_run.h" 
#include <stdio.h>
#include <string.h>
#include <nvml.h>
// #include "results.h"
// #include "debug.h"

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
char deviceNameStr[128];

void nvml_setup(int device, int gr_clock, int mem_clock){

	// run the nvml Init phase
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML init fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	// get the Device ID string for NVML
	nvmlResult =  nvmlDeviceGetHandleByIndex(device, &nvmlDeviceID);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML get Device ID fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr)/sizeof(deviceNameStr[0]));
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML get Device name fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }



	//set the desired min and max GPU clock
	unsigned int gpu_clock;
	gpu_clock = (unsigned int)(gr_clock); //assign_clock(deviceNameStr);
	nvmlResult = nvmlDeviceSetGpuLockedClocks(nvmlDeviceID, gpu_clock, gpu_clock);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML set GPU clock fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	//set the desired min and max MEM clock
	unsigned int MEM_clock;
	MEM_clock = (unsigned int)(mem_clock);
	// nvmlDeviceSetMemoryLockedClocks
	// nvmlDeviceSetMemoryLockedClocks
	nvmlResult = nvmlDeviceSetMemoryLockedClocks(nvmlDeviceID, MEM_clock, MEM_clock);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML set MEM clock fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

}

void nvml_reset(){

	nvmlResult = nvmlDeviceResetMemoryLockedClocks(nvmlDeviceID);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML reset Memory fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	nvmlResult = nvmlDeviceResetGpuLockedClocks(nvmlDeviceID);
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML reset GPU fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

	nvmlResult = nvmlShutdown();
	if (NVML_SUCCESS != nvmlResult){
                printf("NVML shutdown fail: %s\n", nvmlErrorString(nvmlResult));
                exit(0);
        }

}

int read_int(const char* path){
	int num;
	FILE *fptr;

	if ((fptr = fopen(path, "r")) == NULL) {
		printf("Error! opening file");
		exit(1);
	}

	fscanf(fptr, "%d", &num);
	fclose(fptr);
	return num;
}

int main() {
	//Needs to be set
	int device = 1;
	const char* gr_path = "/home/emily/resnet/Resnet/current_clocks/gr.txt";
	const char* mem_path = "/home/emily/resnet/Resnet/current_clocks/mem.txt";

	//Read in device, gr_clock & mem_clock
	int gr_clock, mem_clock;
	gr_clock = read_int(gr_path);
	mem_clock = read_int(mem_path);
	printf("Graphics clock: %d MHz\t Memory clock: %d MHz\n", gr_clock, mem_clock);

	nvml_setup(device, gr_clock, mem_clock);
	// nvml_reset();

	return 0;
}