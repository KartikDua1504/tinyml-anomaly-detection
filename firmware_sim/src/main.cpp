#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <Wire.h>

#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 1. INCLUDE THE MODEL DIRECTLY
// This gives us access to 'fan_low_model' and 'fan_low_model_len'
#include "model.h"

// Threshold (From your Python training)
const float ANOMALY_THRESHOLD = 0.00004;

Adafruit_MPU6050 mpu;

// TFLite Globals
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // 2. LOAD MODEL (Using the name we found in grep)
  model = tflite::GetModel(fan_low_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Schema mismatch!");
    while (1)
      delay(100);
  }

  // Setup Interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System Ready.");
}

void loop() {
  // 3. COLLECT DATA (128 samples)
  for (int i = 0; i < 128; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Fill input buffer
    input->data.f[i * 3 + 0] = a.acceleration.x;
    input->data.f[i * 3 + 1] = a.acceleration.y;
    input->data.f[i * 3 + 2] = a.acceleration.z;

    delay(10); // Sampling rate ~100Hz
  }

  // 4. RUN INFERENCE
  interpreter->Invoke();

  // 5. CALCULATE ERROR (MSE)
  float mse = 0;
  // Calculate over all data points (128 samples * 3 axes = 384)
  for (int i = 0; i < (128 * 3); i++) {
    float diff = input->data.f[i] - output->data.f[i];
    mse += diff * diff;
  }
  mse /= (128 * 3);

  // 6. PRINT RESULT
  Serial.print("MSE: ");
  Serial.print(mse, 7);
  if (mse > ANOMALY_THRESHOLD) {
    Serial.println(" -> ANOMALY!");
  } else {
    Serial.println(" -> Normal");
  }
}
