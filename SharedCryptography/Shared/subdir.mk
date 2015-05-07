################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Cryptography.cpp \
../SymmetricCryptography.cpp 

CU_SRCS += \
../edu_columbia_gpu11_CuWrapper.cu \
../util.cu 

CU_DEPS += \
./edu_columbia_gpu11_CuWrapper.d \
./util.d 

OBJS += \
./Cryptography.o \
./SymmetricCryptography.o \
./edu_columbia_gpu11_CuWrapper.o \
./util.o 

CPP_DEPS += \
./Cryptography.d \
./SymmetricCryptography.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux -G -g -O0 --use_fast_math -Xcompiler -fPIC -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux -G -g -O0 --use_fast_math -Xcompiler -fPIC --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux -G -g -O0 --use_fast_math -Xcompiler -fPIC -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/linux -G -g -O0 --use_fast_math -Xcompiler -fPIC --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


