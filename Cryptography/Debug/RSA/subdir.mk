################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../RSA/CpuRSA.cu \
../RSA/GpuRSA.cu \
../RSA/RSA.cu 

CU_DEPS += \
./RSA/CpuRSA.d \
./RSA/GpuRSA.d \
./RSA/RSA.d 

OBJS += \
./RSA/CpuRSA.o \
./RSA/GpuRSA.o \
./RSA/RSA.o 


# Each subdirectory must supply rules for building sources it contributes
RSA/%.o: ../RSA/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 --use_fast_math -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "RSA" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 --use_fast_math --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


