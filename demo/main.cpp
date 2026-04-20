/*
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file main.cpp
 * @brief Main application entry point for accelerometer inferencing using Edge Impulse SDK on Zephyr.
 */

// --- Zephyr RTOS includes ---
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>
#include <string.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <stdio.h>
#include <zephyr/drivers/display.h>
#include <lvgl.h>

// --- Edge Impulse SDK includes ---
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

// --- Constants and Macros ---
#define STANDARD_GRAVITY 9.80665f

// Sampling parameters
// Note: This frequency should match the data collection frequency in Edge Impulse Studio.
#define SAMPLING_RATE_HZ            100
#define SAMPLING_PERIOD_MS          (1000 / SAMPLING_RATE_HZ)

// Compile-time check for expected sensor axis count
#if EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3
#error "This implementation assumes 3-axis accelerometer data."
#endif

// Calculate the number of samples needed per inference window
#define NUM_SAMPLES_PER_INFERENCE   (EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME)

// Sliding window parameters (1000ms overlap)
#define SLIDING_WINDOW_MS           1000
#define NUM_SAMPLES_TO_SLIDE        ((SLIDING_WINDOW_MS * SAMPLING_RATE_HZ) / 1000)

// Register the logging module
LOG_MODULE_REGISTER(main, LOG_LEVEL_DBG);

// --- Global Variables and Device Definitions ---

// Static buffer to hold the features for one inference window
static float features_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
// Buffer to hold data specifically for the inference thread (Double Buffering)
static float inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
// Variables to track the logical time window of the current inference buffer
static float inference_window_start_s = 0.0f;
static float inference_window_end_s = 0.0f;

// Semaphore to trigger the inference thread
K_SEM_DEFINE(infer_sem, 0, 1);

// --- GUI Variables ---
static char current_face_str[32] = "unknown";
static char current_result_str[128] = "Waiting for AI...";
static K_MUTEX_DEFINE(gui_mutex);
static lv_obj_t *left_eye = nullptr;
static lv_obj_t *right_eye = nullptr;
static lv_obj_t *mouth = nullptr;
static lv_obj_t *result_label = nullptr;
static lv_obj_t *deco_label = nullptr;
static lv_obj_t *progress_bar = nullptr;
static uint32_t inference_count = 0;

// --- Function Prototypes ---
static int init_accel_sensor(const struct device *dev);
static void collect_sensor_data(const struct device *dev, uint32_t num_samples, uint32_t start_index);
static void run_inference(void);
static int raw_feature_get_data(size_t offset, size_t length, float *out_ptr);

// --- Thread Functions ---
static void sensor_thread_fn(void *p1, void *p2, void *p3)
{
    const struct device *const accel_sensor = DEVICE_DT_GET(DT_ALIAS(accel0));

    k_sleep(K_MSEC(1000)); // Give some time for serial terminal to catch up

    if (init_accel_sensor(accel_sensor) != 0) {
        LOG_ERR("Sensor setup failed. Please check wiring and I2C address.");
        return;
    }

    bool first_run = true;
    uint32_t window_count = 0;
    while (1) {
        if (first_run) {
            // Collect data for the entire buffer (2000ms) on the first run
            collect_sensor_data(accel_sensor, NUM_SAMPLES_PER_INFERENCE, 0);
            first_run = false;
        } else {
            // Shift old data to the left (discarding 1000ms) and collect new data into the second half (1000ms)
            size_t shift_elements = NUM_SAMPLES_TO_SLIDE * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
            size_t keep_elements = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - shift_elements;
            memmove(features_buffer, features_buffer + shift_elements, keep_elements * sizeof(float));
            collect_sensor_data(accel_sensor, NUM_SAMPLES_TO_SLIDE, NUM_SAMPLES_PER_INFERENCE - NUM_SAMPLES_TO_SLIDE);
        }

        // Copy to the inference-specific buffer to prevent data from being overwritten during inference
        memcpy(inference_buffer, features_buffer, sizeof(features_buffer));

        // Record the logical time window for this inference (in seconds)
        inference_window_start_s = window_count * (SLIDING_WINDOW_MS / 1000.0f);
        inference_window_end_s = inference_window_start_s + (NUM_SAMPLES_PER_INFERENCE / (float)SAMPLING_RATE_HZ);
        window_count++;

        // Notify the inference thread
        k_sem_give(&infer_sem);
    }
}

static void inference_thread_fn(void *p1, void *p2, void *p3)
{
    while (1) {
        // Wait for data to be ready from the sensor thread
        k_sem_take(&infer_sem, K_FOREVER);
        
        run_inference();
    }
}

// --- GUI Thread ---
static void gui_thread_fn(void *p1, void *p2, void *p3)
{
    const struct device *display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
    if (!device_is_ready(display_dev)) {
        LOG_ERR("Display device not ready. GUI disabled.");
        return;
    }

    // Disable display blanking and turn on the backlight
    display_blanking_off(display_dev);

    // Set screen background to black
    lv_obj_set_style_bg_color(lv_scr_act(), lv_color_black(), 0);

    // --- Create Stackchan-like Face Shapes ---
    left_eye = lv_obj_create(lv_scr_act());
    lv_obj_remove_style_all(left_eye);
    lv_obj_set_style_bg_color(left_eye, lv_color_white(), 0);
    lv_obj_set_style_bg_opa(left_eye, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(left_eye, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_size(left_eye, 30, 40);
    lv_obj_align(left_eye, LV_ALIGN_CENTER, -50, -40);

    right_eye = lv_obj_create(lv_scr_act());
    lv_obj_remove_style_all(right_eye);
    lv_obj_set_style_bg_color(right_eye, lv_color_white(), 0);
    lv_obj_set_style_bg_opa(right_eye, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(right_eye, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_size(right_eye, 30, 40);
    lv_obj_align(right_eye, LV_ALIGN_CENTER, 50, -40);

    mouth = lv_obj_create(lv_scr_act());
    lv_obj_remove_style_all(mouth);
    lv_obj_set_style_bg_color(mouth, lv_color_white(), 0);
    lv_obj_set_style_bg_opa(mouth, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(mouth, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_size(mouth, 40, 10);
    lv_obj_align(mouth, LV_ALIGN_CENTER, 0, 20);

    // Create a label for the results
    result_label = lv_label_create(lv_scr_act());
    lv_obj_set_style_text_color(result_label, lv_color_white(), 0); // Set text color to white
    lv_label_set_text(result_label, current_result_str);
#ifdef CONFIG_LV_FONT_MONTSERRAT_20
    lv_obj_set_style_text_font(result_label, &lv_font_montserrat_20, LV_PART_MAIN);
#elif defined(CONFIG_LV_FONT_MONTSERRAT_14)
    lv_obj_set_style_text_font(result_label, &lv_font_montserrat_14, LV_PART_MAIN);
#endif
    lv_obj_set_style_text_align(result_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(result_label, LV_ALIGN_BOTTOM_MID, 0, -10); // Move label to the bottom

    // Create a label for decorative text (zzz, !!!, www)
    deco_label = lv_label_create(lv_scr_act());
    lv_obj_set_style_text_color(deco_label, lv_color_white(), 0);
    lv_label_set_text(deco_label, "");
#ifdef CONFIG_LV_FONT_MONTSERRAT_32
    lv_obj_set_style_text_font(deco_label, &lv_font_montserrat_32, LV_PART_MAIN);
#elif defined(CONFIG_LV_FONT_MONTSERRAT_20)
    lv_obj_set_style_text_font(deco_label, &lv_font_montserrat_20, LV_PART_MAIN);
#endif
    lv_obj_align(deco_label, LV_ALIGN_CENTER, 100, -70);

    // Create a progress bar at the top of the screen
    progress_bar = lv_bar_create(lv_scr_act());
    lv_obj_set_size(progress_bar, 280, 10);
    lv_obj_align(progress_bar, LV_ALIGN_TOP_MID, 0, 10);
    lv_bar_set_range(progress_bar, 0, SLIDING_WINDOW_MS); // 0 to 1000ms
    lv_obj_set_style_bg_color(progress_bar, lv_color_hex(0x333333), LV_PART_MAIN); // Dark gray background
    lv_obj_set_style_bg_color(progress_bar, lv_color_white(), LV_PART_INDICATOR);  // White indicator
    lv_bar_set_value(progress_bar, 0, LV_ANIM_OFF);

    char local_face_str[32] = "";
    char local_str[128] = "";
    uint32_t local_inference_count = 0;
    int64_t last_reset_time = k_uptime_get();
    int last_anim_step = -1; // To track animation state

    while (1) {
        bool new_result = false;
        
        k_mutex_lock(&gui_mutex, K_FOREVER);
        if (strcmp(local_face_str, current_face_str) != 0) {
            strcpy(local_face_str, current_face_str);
            
            // Update face expressions based on state
            if (strcmp(local_face_str, "m5_idle") == 0) {
                // Sleepy: Eyes closed (horizontal lines), lowered
                lv_obj_set_size(left_eye, 40, 6);
                lv_obj_align(left_eye, LV_ALIGN_CENTER, -50, -25);
                lv_obj_set_size(right_eye, 40, 6);
                lv_obj_align(right_eye, LV_ALIGN_CENTER, 50, -25);
                lv_obj_set_size(mouth, 15, 6);
                lv_obj_align(mouth, LV_ALIGN_CENTER, 0, 35);
            } else if (strcmp(local_face_str, "m5_flick") == 0) {
                // Flick/Action: Squinting, close together, wide shouting mouth
                lv_obj_set_size(left_eye, 25, 15);
                lv_obj_align(left_eye, LV_ALIGN_CENTER, -35, -35);
                lv_obj_set_size(right_eye, 25, 15);
                lv_obj_align(right_eye, LV_ALIGN_CENTER, 35, -35);
                lv_obj_set_size(mouth, 60, 15);
                lv_obj_align(mouth, LV_ALIGN_CENTER, 0, 15);
                lv_label_set_text(deco_label, "");
            } else if (strcmp(local_face_str, "m5_updown") == 0) {
                // Updown: Eyes popping out vertically like a cartoon, tiny 'o' mouth
                lv_obj_set_size(left_eye, 15, 75);
                lv_obj_align(left_eye, LV_ALIGN_CENTER, -45, -65); // Jumped up
                lv_obj_set_size(right_eye, 15, 75);
                lv_obj_align(right_eye, LV_ALIGN_CENTER, 45, -65);
                lv_obj_set_size(mouth, 15, 15);
                lv_obj_align(mouth, LV_ALIGN_CENTER, 0, 25);
                lv_label_set_text(deco_label, "");
            } else if (strcmp(local_face_str, "m5_knock") == 0) {
                // Knock/Shock: Super Surprised! Massive bug eyes, Jaw dropped to the floor
                lv_obj_set_size(left_eye, 65, 65);
                lv_obj_align(left_eye, LV_ALIGN_CENTER, -55, -45); 
                lv_obj_set_size(right_eye, 65, 65);
                lv_obj_align(right_eye, LV_ALIGN_CENTER, 55, -45);
                lv_obj_set_size(mouth, 45, 70); // Very tall mouth
                lv_obj_align(mouth, LV_ALIGN_CENTER, 0, 40); // Dropped way down
                lv_label_set_text(deco_label, "");
            } else {
                // Default/Unknown
                lv_obj_set_size(left_eye, 30, 40);
                lv_obj_align(left_eye, LV_ALIGN_CENTER, -50, -40);
                lv_obj_set_size(right_eye, 30, 40);
                lv_obj_align(right_eye, LV_ALIGN_CENTER, 50, -40);
                lv_obj_set_size(mouth, 40, 10);
                lv_obj_align(mouth, LV_ALIGN_CENTER, 0, 20);
                lv_label_set_text(deco_label, "");
            }
        }

        // Animate decorative text based on the current state
        if (strcmp(local_face_str, "m5_idle") == 0) {
            int anim_step = (k_uptime_get() / 600) % 2; // Slow toggle
            if (anim_step != last_anim_step) {
                lv_label_set_text(deco_label, anim_step == 0 ? "zzz" : "zzz...");
                lv_obj_align(deco_label, LV_ALIGN_CENTER, 100, -60);
                last_anim_step = anim_step;
            }
        } else if (strcmp(local_face_str, "m5_flick") == 0) {
            int anim_step = (k_uptime_get() / 150) % 2; // Fast toggle
            if (anim_step != last_anim_step) {
                lv_label_set_text(deco_label, anim_step == 0 ? "!!" : "!!!");
                lv_obj_align(deco_label, LV_ALIGN_CENTER, 100, -50);
                last_anim_step = anim_step;
            }
        } else if (strcmp(local_face_str, "m5_updown") == 0) {
            int anim_step = (k_uptime_get() / 250) % 2; // Medium toggle
            if (anim_step != last_anim_step) {
                lv_label_set_text(deco_label, anim_step == 0 ? "www" : "wwww");
                lv_obj_align(deco_label, LV_ALIGN_CENTER, 100, -70);
                last_anim_step = anim_step;
            }
        } else if (strcmp(local_face_str, "m5_knock") == 0) {
            int anim_step = (k_uptime_get() / 100) % 2; // Very fast blink (Flash)
            if (anim_step != last_anim_step) {
                lv_label_set_text(deco_label, anim_step == 0 ? "!!!" : "");
                lv_obj_align(deco_label, LV_ALIGN_CENTER, 110, -80);
                last_anim_step = anim_step;
            }
        } else {
            if (last_anim_step != -1) {
                lv_label_set_text(deco_label, "");
                last_anim_step = -1; // Reset animation state
            }
        }

        if (strcmp(local_str, current_result_str) != 0) {
            strcpy(local_str, current_result_str);
            lv_label_set_text(result_label, local_str);
        }
        if (local_inference_count != inference_count) {
            local_inference_count = inference_count;
            new_result = true; // Flag that a new inference result arrived
        }
        k_mutex_unlock(&gui_mutex);

        // Update progress bar
        if (new_result) {
            last_reset_time = k_uptime_get();
        }
        int64_t elapsed = k_uptime_get() - last_reset_time;
        if (elapsed > SLIDING_WINDOW_MS) {
            elapsed = SLIDING_WINDOW_MS;
        }
        lv_bar_set_value(progress_bar, (int32_t)elapsed, LV_ANIM_OFF);

        lv_task_handler();
        k_sleep(K_MSEC(10));
    }
}

// --- Thread Definitions ---
K_THREAD_DEFINE(sensor_tid, 2048, sensor_thread_fn, NULL, NULL, NULL, 5, 0, 0);
K_THREAD_DEFINE(inference_tid, 8192, inference_thread_fn, NULL, NULL, NULL, 7, 0, 0);
// Define GUI thread with priority 6 (between sensor and inference)
K_THREAD_DEFINE(gui_tid, 4096, gui_thread_fn, NULL, NULL, NULL, 6, 0, 0);

// --- Main Application Entry Point ---
extern "C" int main(void)
{
    // Suspend the main thread after spawning worker threads
    k_sleep(K_FOREVER);
    return 0; // Should not be reached
}

// --- Function Definitions ---

/**
 * @brief Initializes the accelerometer sensor.
 */
static int init_accel_sensor(const struct device *dev)
{
    if (!device_is_ready(dev)) {
        LOG_ERR("Accelerometer: Device is not ready.");
        return -1;
    }

    struct sensor_value full_scale, sampling_freq, oversampling;

    /* Setting scale in G, due to loss of precision if the SI unit m/s^2 is used */
    full_scale.val1 = 2;            /* G */
    full_scale.val2 = 0;
    sampling_freq.val1 = SAMPLING_RATE_HZ; /* Hz. Performance mode */
    sampling_freq.val2 = 0;
    oversampling.val1 = 1;          /* Normal mode */
    oversampling.val2 = 0;

    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_FULL_SCALE, &full_scale);
    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_OVERSAMPLING, &oversampling);
    /* Set sampling frequency last as this also sets the appropriate power mode. */
    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

    /* Setting scale in degrees/s to match the sensor scale */
    full_scale.val1 = 500;          /* dps */

    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_FULL_SCALE, &full_scale);
    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_OVERSAMPLING, &oversampling);
    /* Set sampling frequency last as this also sets the appropriate power mode. */
    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

    LOG_INF("Accelerometer: Device is ready.");
    return 0;
}

/**
 * @brief Collects sensor data for one inference window.
 */
static void collect_sensor_data(const struct device *dev, uint32_t num_samples, uint32_t start_index)
{
    struct sensor_value accel[3];

    LOG_INF("Collecting %u samples for the next inference...", num_samples);

    uint32_t samples_read = 0;
    int64_t collection_start_ms = k_uptime_get();

    while (samples_read < num_samples) {
        // Record the start time of the sample to maintain the sampling rate
        int64_t start_time_ms = k_uptime_get();

        // Fetch the latest sample from the sensor
        if (sensor_sample_fetch(dev) == 0) {
            sensor_channel_get(dev, SENSOR_CHAN_ACCEL_XYZ, accel);

            size_t current_index = (start_index + samples_read) * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
            features_buffer[current_index + 0] = (float)sensor_value_to_double(&accel[0]);
            features_buffer[current_index + 1] = (float)sensor_value_to_double(&accel[1]);
            features_buffer[current_index + 2] = (float)sensor_value_to_double(&accel[2]);

            samples_read++;
        } else {
            LOG_WRN("Failed to read sensor data, retrying sample %u...", samples_read);
        }

        // Calculate how long to sleep to match the target sampling period
        int sleep_time_ms = SAMPLING_PERIOD_MS - (int)(k_uptime_get() - start_time_ms);
        if (sleep_time_ms > 0) {
            k_sleep(K_MSEC(sleep_time_ms));
        }
    }
    
    int64_t collection_end_ms = k_uptime_get();
    float actual_period_ms = (float)(collection_end_ms - collection_start_ms) / num_samples;
    float actual_rate_hz = actual_period_ms > 0 ? (1000.0f / actual_period_ms) : 0.0f;
    LOG_INF("Data collection complete. Actual avg rate: %.2f Hz (Target: %d Hz)", (double)actual_rate_hz, SAMPLING_RATE_HZ);
}

/**
 * @brief Runs inference on the collected data and prints the results.
 */
static void run_inference(void)
{
    static int64_t last_inference_time_ms = 0;
    int64_t current_time_ms = k_uptime_get();
    int64_t inference_period_ms = 0;

    if (last_inference_time_ms != 0) {
        inference_period_ms = current_time_ms - last_inference_time_ms;
    }
    last_inference_time_ms = current_time_ms;

    // Create a signal object to wrap the data buffer
    signal_t features_signal;
    features_signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    features_signal.get_data = &raw_feature_get_data;

    ei_impulse_result_t result = {0};

    if (inference_period_ms > 0) {
        LOG_INF("--- Running inference on data window: %.1fs - %.1fs (Period: %lld ms) ---", inference_window_start_s, inference_window_end_s, inference_period_ms);
    } else {
        LOG_INF("--- Running inference on data window: %.1fs - %.1fs ---", inference_window_start_s, inference_window_end_s);
    }

    // Run the classifier (DSP + Neural Network)
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);

    if (res != EI_IMPULSE_OK) {
        LOG_ERR("Classifier returned error: %d", res);
        return;
    }

    LOG_INF("Predictions (DSP: %d ms, Classification: %d ms, Anomaly: %d ms):",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
            
    float max_value = 0.0f;
    const char *best_label = "Unknown";

    // Print the probability for each class
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        float value = (float)result.classification[ix].value;
        LOG_INF("    %s: %.5f", result.classification[ix].label, (double)value);
        
        if (value > max_value) {
            max_value = value;
            best_label = result.classification[ix].label;
        }
    }

    // Update GUI string safely using mutex
    k_mutex_lock(&gui_mutex, K_FOREVER);
    // Pass the state directly instead of an ASCII emoticon
    strcpy(current_face_str, best_label);
    if (inference_period_ms > 0) {
        snprintf(current_result_str, sizeof(current_result_str), "%s : %.0f%%\nPeriod: %lld ms", best_label, (double)(max_value * 100.0f), inference_period_ms);
    } else {
        snprintf(current_result_str, sizeof(current_result_str), "%s : %.0f%%\nPeriod: --- ms", best_label, (double)(max_value * 100.0f));
    }
    inference_count++;
    k_mutex_unlock(&gui_mutex);

#if EI_CLASSIFIER_HAS_ANOMALY == 1
    LOG_INF("    anomaly score: %.3f", (double)result.anomaly);
#endif

    printf("\n"); // Insert an empty line between the 1-second inference logs
}

/**
 * @brief Callback function to provide raw sample data to the Edge Impulse SDK.
 *
 * This function is called by the classifier to get slices of the feature buffer.
 *
 * @param offset Starting index of the data to fetch.
 * @param length Number of float values to fetch.
 * @param out_ptr Pointer to the buffer where the data should be copied.
 * @return 0 on success.
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
    // Copy the requested data from the inference buffer to the SDK's buffer
    memcpy(out_ptr, inference_buffer + offset, length * sizeof(float));
    return 0;
}