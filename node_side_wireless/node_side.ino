/*
  node_side.ino — ESP32 IMU node for wearable shoulder project
  -------------------------------------------------------------
  • Reads MPU9250 over I2C (hideakitai/MPU9250 API)
  • 6-axis or 9-axis Madgwick fusion (mag only for chest node)
  • Fixed-rate fusion loop (FUSION_HZ) + downsampled ESP-NOW send (SEND_HZ)
  • On-command gyro bias calibration while node is kept still
  • Simple quaternion low-pass for visualization
  • ESP-NOW packet: {magic, version, nodeId, seq, t_node_us, q[4], acc[3], gyr[3], flags}

  NOTE:
    - For calibration: send CMD_CALIBRATE, keep the node STILL for ~2 seconds.
      For anatomical zero (joint angles), we'll use N-pose on the Python/Unity side.
*/

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#include <MadgwickAHRS.h>
#include <MPU9250.h>   // hideakitai MPU9250

// ====== ---------- CONFIG ---------- ======

// Per-node configuration
#define NODE_ID           2            // <-- set a unique ID (1=wrist,2=forearm,3=upper,4=chest, etc.)
#define IS_CHEST_NODE     0            // 1 for chest node, 0 for other limbs
#define USE_MAG           (IS_CHEST_NODE)  // only chest uses magnetometer in fusion

// Wireless
#define WIFI_CHANNEL      1            // must match controller

// Put your controller's station MAC address here:
static uint8_t MASTER_MAC[6] = {0x80, 0xB5, 0x4E, 0xC1, 0xB4, 0xBC};  //  set  MAC

// IMU settings
#define FUSION_HZ         200          // sensor fusion rate (Hz)
#define SEND_HZ           100          // radio send rate (Hz) <= FUSION_HZ
#define I2C_SDA_PIN       8            // adjust to your wiring
#define I2C_SCL_PIN       9
#define I2C_FREQ_HZ       400000

#define IMU_ADDR_PRIMARY  0x68
#define IMU_ADDR_ALT      0x69

// Madgwick tuning
#define MADGWICK_BETA     0.1f         // fusion gain (tune if needed)

// Magnetometer gating (only for chest node)
#define MAG_FIELD_NORM_Ut   45.0f      // typical local field strength in µT (rough)
#define MAG_NORM_TOL        0.20f      // ±20% window
#define GYRO_DPS_GATE       90.0f      // only trust mag at low angular velocity (deg/s)

// Quaternion smoothing (for visualization; keeps data stable)
#define ANGLE_LP_ALPHA      0.20f      // 0..1 (higher = less smoothing)

// Calibration settings
#define CALIB_DURATION_S    2.0f       // seconds to collect still data
// We'll use CALIB_SAMPLES = FUSION_HZ * CALIB_DURATION_S

// --- Commands from controller to nodes ---
#pragma pack(push, 1)
struct CtrlCommand {
  uint8_t  magic;    // 0xC3
  uint8_t  version;  // 1
  uint8_t  cmd;      // see enum below
  uint8_t  reserved;
  uint32_t arg;      // optional (duration, etc.), not used for now
};
#pragma pack(pop)

enum CtrlCmd : uint8_t {
  CMD_CALIBRATE    = 1,
  CMD_START_STREAM = 2,
  CMD_STOP_STREAM  = 3,
};

enum NodeState {
  NODE_IDLE = 0,        // waiting for commands
  NODE_CALIBRATING = 1, // in calibration window
  NODE_STREAMING = 2    // running fusion + sending
};

static NodeState nodeState = NODE_IDLE;


// ====== IMU + Fusion objects ======
MPU9250 imu;
Madgwick filter;

// IMU ready flag (for retry logic)
static bool imuReady = false;

// ====== Timing ======
static uint32_t fusionPeriodUs = 1000000UL / FUSION_HZ;
static uint32_t sendPeriodUs   = 1000000UL / SEND_HZ;
static uint32_t lastFusionUs   = 0;
static uint32_t lastSendUs     = 0;

// ====== Calibration state ======
static const int CALIB_SAMPLES = (int)(FUSION_HZ * CALIB_DURATION_S);

static bool   calibDone  = false;
static int    calibCount = 0;
static float  gyroSum[3] = {0,0,0};
static float  accSum[3]  = {0,0,0};
static float  gyroBias[3] = {0,0,0};
static float  accBias[3]  = {0,0,0};   // mean accel (in g); currently not subtracted from fusion

// ====== Fusion state ======
static volatile uint32_t seq = 0;
static float qSmooth[4] = {1,0,0,0};   // smoothed quaternion

// ====== ESP-NOW packet ======
#pragma pack(push, 1)
typedef struct {
  uint8_t  magic;       // 0xA5
  uint8_t  version;     // 1
  uint8_t  nodeId;      // NODE_ID
  uint8_t  reserved;    // for future use

  uint32_t seq;         // packet sequence (per node)
  uint32_t t_node_us;   // esp_timer_get_time() (low 32 bits is fine)

  float    q[4];        // w,x,y,z (mount frame)
  float    acc[3];      // accel (bias-corrected), in g
  float    gyr[3];      // gyro (bias-corrected), in deg/s

  uint8_t  flags;       // bit0 = mag_used, bit1 = imu_ok
} ImuPacket;
#pragma pack(pop)

// ====== Forward declarations ======
bool initImu();
bool imuRead(float acc[3], float gyr[3], float mag[3], bool &magValid);
void runCalibration(float acc[3], float gyr[3]);
void stepFusionAndSend();
bool initEspNow();

// ====== ESP-NOW callbacks ======
void onSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Optional: you can toggle an LED or keep stats here
}

// Node-side ESP-NOW RX: listen for controller commands
static void onCtrlRecv(const esp_now_recv_info *info,
                       const uint8_t *data,
                       int len) {
  if (len < (int)sizeof(CtrlCommand)) return;

  CtrlCommand c;
  memcpy(&c, data, sizeof(CtrlCommand));

  if (c.magic != 0xC3 || c.version != 1) return;

  switch (c.cmd) {
    case CMD_CALIBRATE:
      Serial.println("[CMD] CALIBRATE received");

      // reset calibration accumulators
      calibDone  = false;
      calibCount = 0;
      gyroSum[0] = gyroSum[1] = gyroSum[2] = 0.0f;
      accSum[0]  = accSum[1]  = accSum[2]  = 0.0f;

      nodeState  = NODE_CALIBRATING;
      break;

    case CMD_START_STREAM:
      Serial.println("[CMD] START_STREAM received");
      if (calibDone) {
        nodeState = NODE_STREAMING;
      } else {
        Serial.println("[CMD] Ignored START_STREAM (calib not done)");
      }
      break;

    case CMD_STOP_STREAM:
      Serial.println("[CMD] STOP_STREAM received");
      nodeState = NODE_IDLE;
      break;

    default:
      break;
  }
}


// ====== SETUP ======
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== IMU NODE BOOT ===");

  // I2C
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ_HZ);

  imuReady = initImu();
  if (!imuReady) {
    Serial.println("[IMU] init failed, will keep retrying in loop");
  }

  // Madgwick
  filter.begin(FUSION_HZ);
  // If you want: filter.setBeta(MADGWICK_BETA);

  // ESP-NOW init
  if (!initEspNow()) {
    Serial.println("[ESP-NOW] init failed");
  }

  lastFusionUs = micros();
  lastSendUs   = lastFusionUs;

  Serial.println("[NODE] Waiting in IDLE. Send CALIBRATE command from controller before each recording.");
  nodeState = NODE_IDLE;
  calibDone = false;
}

// ====== LOOP ======
void loop() {
  uint32_t now = micros();

  // Fixed-rate fusion loop (approx)
  if ((uint32_t)(now - lastFusionUs) >= fusionPeriodUs) {
    lastFusionUs += fusionPeriodUs;
    stepFusionAndSend();
  }

  // (You can add low-priority background tasks here if needed)
}

// ====== IMU INIT ======
bool initImu() {
  // Ensure I2C is started
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ_HZ);
  delay(10);

  // Configure MPU9250 using hideakitai's setting struct
  MPU9250Setting setting;
  setting.accel_fs_sel     = ACCEL_FS_SEL::A4G;          // ±4 g
  setting.gyro_fs_sel      = GYRO_FS_SEL::G1000DPS;      // ±1000 °/s
  setting.mag_output_bits  = MAG_OUTPUT_BITS::M16BITS;   // 16-bit mag
  setting.fifo_sample_rate = FIFO_SAMPLE_RATE::SMPL_200HZ;
  setting.gyro_fchoice     = 0x03;
  setting.gyro_dlpf_cfg    = GYRO_DLPF_CFG::DLPF_41HZ;   // ~41 Hz gyro LPF
  setting.accel_fchoice    = 0x01;
  setting.accel_dlpf_cfg   = ACCEL_DLPF_CFG::DLPF_45HZ;  // ~45 Hz accel LPF

  bool ok = imu.setup(IMU_ADDR_PRIMARY, setting);
  if (!ok) {
    Serial.println("[IMU] primary addr 0x68 failed, trying alt 0x69");
    ok = imu.setup(IMU_ADDR_ALT, setting);
    if (!ok) {
      Serial.println("[IMU] setup failed on both addresses");
      return false;
    }
  }

  // Enable internal AHRS (debug/backup) — we use our own Madgwick instance
  imu.ahrs(true);
  imu.selectFilter(QuatFilterSel::MADGWICK);
  imu.setFilterIterations(10);   // 10–20 is typical for stable yaw

  Serial.println("[IMU] ready (use CMD_CALIBRATE before streaming)");
  return true;
}

// ====== IMU READ ======
bool imuRead(float acc[3], float gyr[3], float mag[3], bool &magValid) {
  // Returns true on success; false if no new sample is ready yet
  if (!imu.update()) {
    magValid = false;
    return false;
  }

  // Accel: leave in g units (as returned by library)
  acc[0] = imu.getAccX();
  acc[1] = imu.getAccY();
  acc[2] = imu.getAccZ();

  // Gyro: deg/s
  gyr[0] = imu.getGyroX();
  gyr[1] = imu.getGyroY();
  gyr[2] = imu.getGyroZ();

#if USE_MAG
  // Magnetometer: µT
  mag[0] = imu.getMagX();
  mag[1] = imu.getMagY();
  mag[2] = imu.getMagZ();

  // Gating based on field norm
  float magNorm = sqrtf(mag[0]*mag[0] + mag[1]*mag[1] + mag[2]*mag[2]);
  float relErr  = fabsf(magNorm - MAG_FIELD_NORM_Ut) / MAG_FIELD_NORM_Ut;
  magValid = (relErr <= MAG_NORM_TOL);
#else
  mag[0] = mag[1] = mag[2] = 0.0f;
  magValid = false;
#endif

  return true;
}

// ====== Calibration ======
void runCalibration(float acc[3], float gyr[3]) {
  if (calibCount == 0) {
    Serial.println("[CAL] Starting gyro+accel bias calibration. KEEP NODE PERFECTLY STILL.");
  }

  gyroSum[0] += gyr[0]; gyroSum[1] += gyr[1]; gyroSum[2] += gyr[2];
  accSum[0]  += acc[0]; accSum[1]  += acc[1]; accSum[2]  += acc[2];
  calibCount++;

  if (calibCount >= CALIB_SAMPLES) {
    float invN = 1.0f / (float)CALIB_SAMPLES;

    gyroBias[0] = gyroSum[0] * invN;
    gyroBias[1] = gyroSum[1] * invN;
    gyroBias[2] = gyroSum[2] * invN;

    accBias[0] = accSum[0] * invN;
    accBias[1] = accSum[1] * invN;
    accBias[2] = accSum[2] * invN;

    calibDone = true;

    Serial.printf("[CAL] gyroBias = (%.4f, %.4f, %.4f) deg/s\n",
                  gyroBias[0], gyroBias[1], gyroBias[2]);
    Serial.printf("[CAL] mean accel ~ (%.4f, %.4f, %.4f) g\n",
                  accBias[0], accBias[1], accBias[2]);
    Serial.println("[CAL] Completed.");
  }
}

// ====== Fusion + send ======
void stepFusionAndSend() {
  float acc[3], gyr[3], mag[3];
  bool magValid = false;

  // Ensure IMU initialized (retry if needed)
  if (!imuReady) {
    imuReady = initImu();
    if (!imuReady) {
      static uint32_t lastPrintMs = 0;
      uint32_t nowMs = millis();
      if (nowMs - lastPrintMs > 1000) {
        Serial.println("[IMU] init failed, retrying...");
        lastPrintMs = nowMs;
      }
      return;
    }
  }

  // If we're idle, we don't even read the IMU
  if (nodeState == NODE_IDLE) {
    return;
  }

  // Read IMU
  if (!imuRead(acc, gyr, mag, magValid)) {
    return;
  }

  // 1) Calibration mode
  if (nodeState == NODE_CALIBRATING) {
    runCalibration(acc, gyr);   // will set calibDone when ready
    if (calibDone) {
      nodeState = NODE_IDLE;    // wait for explicit START_STREAM command
      Serial.println("[CAL] Done, back to IDLE (waiting for START_STREAM)");
    }
    return; // do not fuse or send while calibrating
  }

  // 2) Streaming mode (only if calibrated)
  if (nodeState != NODE_STREAMING || !calibDone) {
    return;  // safety: no fusion until we have calibration and START_STREAM
  }

  // ---- Fusion + send ----

  // Bias-correct gyro only; keep accel containing gravity for Madgwick
  gyr[0] -= gyroBias[0];
  gyr[1] -= gyroBias[1];
  gyr[2] -= gyroBias[2];

  // Convert gyro deg/s -> rad/s for Madgwick
  const float DEG2RAD = PI / 180.0f;
  float gx = gyr[0] * DEG2RAD;
  float gy = gyr[1] * DEG2RAD;
  float gz = gyr[2] * DEG2RAD;

  bool usedMag = false;

  // Gate magnetometer on speed + norm
  float gyroNormDps = sqrtf(gyr[0]*gyr[0] + gyr[1]*gyr[1] + gyr[2]*gyr[2]);

  if (USE_MAG && magValid && gyroNormDps < GYRO_DPS_GATE) {
    filter.update(gx, gy, gz,
                  acc[0], acc[1], acc[2],
                  mag[0], mag[1], mag[2]);
    usedMag = true;
  } else {
    filter.updateIMU(gx, gy, gz,
                     acc[0], acc[1], acc[2]);
    usedMag = false;
  }

  // Get orientation as Euler angles (degrees) from Madgwick
  float rollDeg  = filter.getRoll();   // rotation about X
  float pitchDeg = filter.getPitch();  // rotation about Y
  float yawDeg   = filter.getYaw();    // rotation about Z

  // Convert Euler -> quaternion (Z-Y-X / yaw-pitch-roll)
  float halfRoll  = (rollDeg  * DEG2RAD) * 0.5f;
  float halfPitch = (pitchDeg * DEG2RAD) * 0.5f;
  float halfYaw   = (yawDeg   * DEG2RAD) * 0.5f;

  float cr = cosf(halfRoll);
  float sr = sinf(halfRoll);
  float cp = cosf(halfPitch);
  float sp = sinf(halfPitch);
  float cy = cosf(halfYaw);
  float sy = sinf(halfYaw);

  float qw = cr*cp*cy + sr*sp*sy;
  float qx = sr*cp*cy - cr*sp*sy;
  float qy = cr*sp*cy + sr*cp*sy;
  float qz = cr*cp*sy - sr*sp*cy;

  // Smooth quaternion (simple exponential smoothing)
  qSmooth[0] = (1.0f - ANGLE_LP_ALPHA) * qSmooth[0] + ANGLE_LP_ALPHA * qw;
  qSmooth[1] = (1.0f - ANGLE_LP_ALPHA) * qSmooth[1] + ANGLE_LP_ALPHA * qx;
  qSmooth[2] = (1.0f - ANGLE_LP_ALPHA) * qSmooth[2] + ANGLE_LP_ALPHA * qy;
  qSmooth[3] = (1.0f - ANGLE_LP_ALPHA) * qSmooth[3] + ANGLE_LP_ALPHA * qz;

  // Re-normalize smoothed quaternion
  float nq = sqrtf(qSmooth[0]*qSmooth[0] + qSmooth[1]*qSmooth[1] +
                   qSmooth[2]*qSmooth[2] + qSmooth[3]*qSmooth[3]);
  if (nq > 0.0f) {
    float inv = 1.0f / nq;
    qSmooth[0] *= inv;
    qSmooth[1] *= inv;
    qSmooth[2] *= inv;
    qSmooth[3] *= inv;
  } else {
    // in case of NaN or 0, reset
    qSmooth[0] = 1; qSmooth[1] = qSmooth[2] = qSmooth[3] = 0;
  }

  // Send at SEND_HZ
  uint32_t now = micros();
  if ((uint32_t)(now - lastSendUs) >= sendPeriodUs) {
    lastSendUs += sendPeriodUs;

    ImuPacket pkt;
    pkt.magic     = 0xA5;
    pkt.version   = 1;
    pkt.nodeId    = NODE_ID;
    pkt.reserved  = 0;
    pkt.seq       = seq++;
    pkt.t_node_us = (uint32_t)esp_timer_get_time();

    pkt.q[0] = qSmooth[0];
    pkt.q[1] = qSmooth[1];
    pkt.q[2] = qSmooth[2];
    pkt.q[3] = qSmooth[3];

    // We send gyro-bias-corrected accel & gyro
    pkt.acc[0] = acc[0];
    pkt.acc[1] = acc[1];
    pkt.acc[2] = acc[2];

    pkt.gyr[0] = gyr[0];
    pkt.gyr[1] = gyr[1];
    pkt.gyr[2] = gyr[2];

    pkt.flags  = 0;
    if (usedMag) pkt.flags |= 0x01;
    pkt.flags |= 0x02; // imu_ok

    esp_err_t e = esp_now_send(MASTER_MAC, (uint8_t*)&pkt, sizeof(pkt));
    if (e != ESP_OK) {
      Serial.printf("[ESP-NOW] send err=%d\n", e);
    }
  }
}

// ====== ESP-NOW init ======
bool initEspNow() {
  WiFi.mode(WIFI_STA);
  // lock to channel (so nodes+controller all share)
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] init failed");
    return false;
  }

  esp_now_register_send_cb(onSent);
  esp_now_register_recv_cb(onCtrlRecv);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, MASTER_MAC, 6);
  peerInfo.channel = WIFI_CHANNEL;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("[ESP-NOW] add_peer failed");
    return false;
  }

  Serial.println("[ESP-NOW] ready");
  return true;
}
