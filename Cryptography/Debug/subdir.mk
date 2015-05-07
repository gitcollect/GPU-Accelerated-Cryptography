################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Cryptography.cpp \
../SymmetricCryptography.cpp 

CU_SRCS += \
../main.cu \
../util.cu 

CU_DEPS += \
./main.d \
./util.d 

OBJS += \
./Cryptography.o \
./SymmetricCryptography.o \
./main.o \
./util.o 

CPP_DEPS += \
./Cryptography.d \
./SymmetricCryptography.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 --use_fast_math -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 --use_fast_math --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 --use_fast_math -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 --use_fast_math --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


