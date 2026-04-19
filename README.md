# EdgeAI IMU Real Inference (M5Stack CoreS3)

This repository is a project that executes EdgeAI inference based on accelerometer (BMI270) data using the M5Stack CoreS3 running Zephyr RTOS.

## Overview

Motion data acquired from the accelerometer is processed in real-time, and inference is performed by an AI model on the edge device.

**Demo Video:** [https://youtu.be/h2TyvIFuZGY](https://youtu.be/h2TyvIFuZGY)

## Features

*   **Multi-threading with Zephyr RTOS**:
    The application utilizes Zephyr's robust thread management to cleanly separate tasks and maintain strict timing:

    *   **Sensor Thread (Priority 5, High)**: Ensures reliable 100Hz (10ms) data sampling from the BMI270 accelerometer without dropping any frames.
    *   **GUI Thread (Priority 6, Medium)**: Handles asynchronous UI updates on the display.
    *   **Inference Thread (Priority 7, Low)**: Executes the computationally heavy Edge Impulse classification model in the background.

*   **Continuous Inference & Double Buffering**:
    Implements a sliding window approach (e.g., a 2000ms window sliding every 1000ms) alongside double buffering.
    
    This allows the AI to output predictions every second smoothly, while the sensor continues to fill the primary buffer without being blocked by the inference process.

*   **Interactive GUI (LVGL)**:
    Uses the **LVGL** (Light and Versatile Graphics Library) supported by Zephyr to show real-time inference results on the M5Stack CoreS3's LCD screen.
    
    It dynamically changes expressive ASCII emoticons based on the recognized gestures:

    *   **Idle**: `( -_- ) zzz`
    *   **Flick**: `( >_< )`
    *   **Updown**: `( @o@ )`
    *   **Knock**: `( O_O )!`

## Architecture

The following timeline illustrates the Zephyr RTOS multi-threaded concurrent processing and double-buffering mechanism.

```text
[ Timeline: RTOS Multi-threaded Concurrent Processing ]

Time(ms)  | 0    10   20  ... 1000  1010  1020 ... 1500                 ... 2000
----------|-------------------------------------------------------------------------
1. Sensor | #    #    #   ...  #(*)  #    #   ...   #                    ... #(*)
(Pri: 5)  | |    |    |        |     |    |         |                      |
 Buffers  |[features_buffer]   |   [ features_buffer (next frame)     ]    |
          |                    +-> [ inference_buffer copy & notify ]      +->
----------|-------------------------------------------------------------------------
2. GUI    |   @    @    @ ...    @    @    @ ...    @ <*Text Updated*> ... @
(Pri: 6)  |   \--(10ms draw)     |                  |                      |
----------|-------------------------------------------------------------------------
3. Infer. | (--- waiting ---)  =  =  =  =  =  ... = (waiting)              =  =
(Pri: 7)  |                    ^                    |                      ^
          |                    \--(read buffer)     \--(Mutex safe update)
```

**Legend:**
*   `#` : Sensor Sampling (100Hz)
*   `@` : LVGL GUI Task Handler (100Hz)
*   `=` : AI Inference Processing (Edge Impulse)
*   `(*)` : Buffer is full. Trigger inference (Semaphore Give)

**Why Zephyr RTOS? (Key Benefits):**
1.  **Preemption (Zero Dropped Samples):** Even when heavy AI inference (`=`) is running, the highest priority Sensor thread (`#`) preempts it every 10ms to safely collect data.
2.  **Event-Driven Sync:** The Inference thread sleeps completely until the Sensor thread signals via a Semaphore `(*)`, saving CPU cycles.
3.  **Thread Safety:** The AI cleanly updates the GUI text using a Mutex to prevent race conditions with the rapidly drawing GUI thread.

## Edge Impulse Model

You need to create and download your own Edge Impulse model for this project.

1.  Create a project on [Edge Impulse](https://www.edgeimpulse.com/).
2.  Collect data and train a model.
3.  Deploy as a **C++ library**.
4.  Download and extract the library content (`edge-impulse-sdk`, `model-parameters`, `tflite-model`) into the root of this repository.

## Directory Structure

```text
.
├── CMakeLists.txt          # Zephyr CMake configuration
├── prj.conf                # Zephyr project configuration
├── src/
│   └── main.cpp            # Main application source code (Threads, LVGL, etc.)
├── edge-impulse-sdk/       # (Requires Download) Edge Impulse C++ SDK
├── model-parameters/       # (Requires Download) Model parameters
└── tflite-model/           # (Requires Download) TensorFlow Lite Micro model
```

## Hardware Requirements

*   **Development Boards**:
    *   M5Stack CoreS3 (ESP32-S3)
*   **Sensor**:
    *   Bosch BMI270 (Internal 6-axis IMU)

## Build and Flash

To build and flash the application for the M5Stack CoreS3, run the following commands:

```bash
west build -p -b m5stack_cores3/esp32s3/procpu
west flash
```

## License

SPDX-License-Identifier: Apache-2.0
