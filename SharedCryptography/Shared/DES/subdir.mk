################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../DES/CpuDES.cu \
../DES/DES.cu \
../DES/GpuDES.cu 

CU_DEPS += \
./DES/CpuDES.d \
./DES/DES.d \
./DES/GpuDES.d 

OBJS += \
./DES/CpuDES.o \
./DES/DES.o \
./DES/GpuDES.o 


# Each subdirectory must supply rules for building sources it contributes
DES/%.o: ../DES/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux -G -g -O0 --use_fast_math -Xcompiler -fPIC -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "DES" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux -G -g -O0 --use_fast_math -Xcompiler -fPIC --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


