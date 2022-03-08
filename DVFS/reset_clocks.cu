#include "nvml_run.h" 
#include <stdio.h>
#include <string.h>
#include <nvml.h>
// #include "results.h"
// #include "debug.h"

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
char deviceNameStr[128];

void nvml_setup(int device){

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


int main() {
	//Needs to be set
	int device = 1;
    
	nvml_setup(device);
	nvml_reset();

	return 0;
}