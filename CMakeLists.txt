# SPDX-License-Identifier: Apache-2.0

# 1. Basic Zephyr project setup
cmake_minimum_required(VERSION 3.20.5)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(imu_real_inference)

# 2. Add the Edge Impulse SDK as a subdirectory.
#    This includes the TFLite model source files and other dependencies.
include(edge-impulse-sdk/cmake/utils.cmake)
add_subdirectory(edge-impulse-sdk/cmake/zephyr)

# 3. Add model source files and the main application source file to the target.
#    In Zephyr, using target_sources(app ...) automatically creates the 'app' target.

#    Recursively find and add the model source files.
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "tflite-model" "*.cpp")
target_sources(app PRIVATE ${MODEL_SOURCE})

#    Add the main source file (main.cpp).
#    If the file is in the 'src' folder:
target_sources(app PRIVATE src/main.cpp)
#    If the file is in the project root directory:
#    target_sources(app PRIVATE main.cpp)

# 4. Specify include directories.
#    Add the project root directory to the include path.
target_include_directories(app PRIVATE .)

# 5. Suppress warnings from Edge Impulse SDK
target_compile_options(app PRIVATE -Wno-double-promotion -Wno-maybe-uninitialized)

# 6. Disable ESP-IDF specific hardware acceleration (ESP-NN and ESP-DSP)
target_compile_definitions(app PRIVATE
    EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0
    EIDSP_USE_ESP_DSP=0
)

# 7. Add missing include directories for ESP-DSP
target_include_directories(app PRIVATE
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/common/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/math/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/matrix/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/fft/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/dct/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/conv/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/iir/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/fir/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/windows/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/support/include
    edge-impulse-sdk/porting/espressif/esp-dsp/modules/dotprod/include
)