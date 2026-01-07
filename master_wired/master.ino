#include <Wire.h>

// ===================== I2C + multiplexer config =====================

#define I2C_SDA      8      // <-- adjust to your ESP32-S3 board if needed
#define I2C_SCL      9
#define I2C_FREQ     400000

#define TCA_ADDR     0x70   // TCA9548A default address
#define IMU_ADDR     0x68   // MPU6050/9250 default (AD0 = GND)

// Magnetometer (AK8963 on MPU-9250) I2C address
#define MAG_ADDR     0x0C

// ===================== IMU registers =====================

#define REG_PWR_MGMT_1    0x6B
#define REG_SMPLRT_DIV    0x19
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C
#define REG_ACCEL_XOUT_H  0x3B
#define REG_WHO_AM_I      0x75
#define REG_INT_PIN_CFG   0x37  // for BYPASS_EN

// AK8963 magnetometer registers
#define AK8963_REG_WHO_AM_I  0x00
#define AK8963_REG_ST1       0x02
#define AK8963_REG_HXL       0x03
#define AK8963_REG_ST2       0x09
#define AK8963_REG_CNTL1     0x0A

// Sensitivities for chosen full-scale ranges:
//   accel ±4g  -> 8192 LSB/g
//   gyro  ±2000 dps -> 16.4 LSB/(°/s)
//   mag  16-bit -> 0.15 µT/LSB (AK8963)
const float ACCEL_SENS = 8192.0f;
const float GYRO_SENS  = 16.4f;
const float MAG_SENS   = 0.15f;   // µT per LSB (approx)

// ===================== Madgwick filter =====================

class Madgwick {
public:
  float beta;
  float q0, q1, q2, q3;

  Madgwick(float beta_init = 0.1f)
    : beta(beta_init), q0(1.0f), q1(0.0f), q2(0.0f), q3(0.0f) {}

  void updateIMU(float gx, float gy, float gz,
                 float ax, float ay, float az,
                 float dt) {
    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;
    float _2q0, _2q1, _2q2, _2q3;
    float _4q0, _4q1, _4q2, _4q3;
    float _8q1, _8q2;
    float q0q0, q1q1, q2q2, q3q3;

    // Normalise accelerometer
    recipNorm = ax * ax + ay * ay + az * az;
    if (recipNorm <= 0.0f) return;
    recipNorm = fastInvSqrt(recipNorm);
    ax *= recipNorm;
    ay *= recipNorm;
    az *= recipNorm;

    _2q0 = 2.0f * q0;
    _2q1 = 2.0f * q1;
    _2q2 = 2.0f * q2;
    _2q3 = 2.0f * q3;
    _4q0 = 4.0f * q0;
    _4q1 = 4.0f * q1;
    _4q2 = 4.0f * q2;
    _4q3 = 4.0f * q3;
    _8q1 = 8.0f * q1;
    _8q2 = 8.0f * q2;
    q0q0 = q0 * q0;
    q1q1 = q1 * q1;
    q2q2 = q2 * q2;
    q3q3 = q3 * q3;

    // Gradient descent step
    s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
    s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1
       + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
    s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2
       + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
    s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;

    recipNorm = fastInvSqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3);
    s0 *= recipNorm;
    s1 *= recipNorm;
    s2 *= recipNorm;
    s3 *= recipNorm;

    // Quaternion rate of change
    qDot1 = 0.5f * (-q1*gx - q2*gy - q3*gz) - beta * s0;
    qDot2 = 0.5f * ( q0*gx + q2*gz - q3*gy) - beta * s1;
    qDot3 = 0.5f * ( q0*gy - q1*gz + q3*gx) - beta * s2;
    qDot4 = 0.5f * ( q0*gz + q1*gy - q2*gx) - beta * s3;

    // Integrate
    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    // Normalise quaternion
    recipNorm = fastInvSqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
  }

  void getQuaternion(float &w, float &x, float &y, float &z) {
    w = q0; x = q1; y = q2; z = q3;
  }

private:
  float fastInvSqrt(float x) {
    float halfx = 0.5f * x;
    float y = x;
    long i = *(long*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y = y * (1.5f - (halfx * y * y));
    return y;
  }
};

// ===================== Bilateral IMU layout =====================

// Total IMUs: chest + right upper + right fore + left upper + left fore = 5
const int N_IMU = 5;

// Map each IMU index to a TCA channel
const uint8_t IMU_CHANNEL[N_IMU] = {
  0, // 0: chest
  1, // 1: shoulder_R
  2, // 2: elbow_R
  3, // 3: shoulder_L
  4  // 4: elbow_L
};

// Node IDs we’ll use in Python side
const int NODE_ID[N_IMU] = {
  4, // chest
  3, // shoulder_R
  2, // elbow_R
  5, // shoulder_L
  6  // elbow_L
};

// Human-readable names (will go into JSON "name" field)
const char* NODE_NAME[N_IMU] = {
  "chest",
  "shoulder_R",
  "elbow_R",
  "shoulder_L",
  "elbow_L"
};

// One Madgwick filter per IMU (you can tune these betas)
Madgwick filters[N_IMU] = {
  Madgwick(0.01f), // chest
  Madgwick(0.03f), // shoulder_R
  Madgwick(0.05f), // elbow_R
  Madgwick(0.03f), // shoulder_L
  Madgwick(0.05f), // elbow_L
};

// Per-IMU sequence counters
uint32_t seq_counter[N_IMU] = {0,0,0,0,0};

// Timing
uint32_t last_update_us = 0;

// ===================== I2C helpers =====================

void tca_select(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);  // only this channel ON
  Wire.endTransmission();
}

bool i2c_write_byte(uint8_t addr, uint8_t reg, uint8_t data) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(data);
  return (Wire.endTransmission() == 0);
}

bool i2c_read_bytes(uint8_t addr, uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  Wire.requestFrom((int)addr, (int)len);
  if ((int)Wire.available() < (int)len) return false;
  for (size_t i = 0; i < len; i++) {
    buf[i] = Wire.read();
  }
  return true;
}

// ===================== IMU init & read =====================

bool imu_init_on_channel(uint8_t channel) {
  tca_select(channel);

  // Wake up device
  if (!i2c_write_byte(IMU_ADDR, REG_PWR_MGMT_1, 0x00)) {
    return false;
  }
  delay(10);

  // Sample rate divider (1 kHz / (1 + 7) = 125 Hz)
  i2c_write_byte(IMU_ADDR, REG_SMPLRT_DIV, 7);

  // Config: DLPF ~44 Hz
  i2c_write_byte(IMU_ADDR, REG_CONFIG, 0x03);

  // Gyro config: ±2000 dps
  i2c_write_byte(IMU_ADDR, REG_GYRO_CONFIG, 0x18);

  // Accel config: ±4g
  i2c_write_byte(IMU_ADDR, REG_ACCEL_CONFIG, 0x08);

  // Check WHO_AM_I
  uint8_t who = 0;
  if (!i2c_read_bytes(IMU_ADDR, REG_WHO_AM_I, &who, 1)) {
    return false;
  }

  Serial.print("[imu_init] ch=");
  Serial.print(channel);
  Serial.print(" WHO_AM_I=0x");
  Serial.println(who, HEX);

  return true;
}

bool imu_read_on_channel(uint8_t channel,
                         float &ax_g, float &ay_g, float &az_g,
                         float &gx_dps, float &gy_dps, float &gz_dps) {
  tca_select(channel);

  uint8_t buf[14];
  if (!i2c_read_bytes(IMU_ADDR, REG_ACCEL_XOUT_H, buf, 14)) {
    return false;
  }

  int16_t ax = (int16_t)((buf[0] << 8) | buf[1]);
  int16_t ay = (int16_t)((buf[2] << 8) | buf[3]);
  int16_t az = (int16_t)((buf[4] << 8) | buf[5]);
  // int16_t temp = (int16_t)((buf[6] << 8) | buf[7]);
  int16_t gx = (int16_t)((buf[8] << 8) | buf[9]);
  int16_t gy = (int16_t)((buf[10] << 8) | buf[11]);
  int16_t gz = (int16_t)((buf[12] << 8) | buf[13]);

  ax_g = (float)ax / ACCEL_SENS;
  ay_g = (float)ay / ACCEL_SENS;
  az_g = (float)az / ACCEL_SENS;

  gx_dps = (float)gx / GYRO_SENS;
  gy_dps = (float)gy / GYRO_SENS;
  gz_dps = (float)gz / GYRO_SENS;

  return true;
}

// ===================== Chest magnetometer helpers =====================

// Enable bypass + initialise AK8963 on the chest IMU (channel 0)
bool mag_init_chest() {
  const uint8_t chest_ch = IMU_CHANNEL[0]; // assume index 0 is chest
  tca_select(chest_ch);

  // Enable bypass so AK8963 is visible on main I2C bus at MAG_ADDR
  // Set BYPASS_EN bit in INT_PIN_CFG (0x37)
  if (!i2c_write_byte(IMU_ADDR, REG_INT_PIN_CFG, 0x02)) {
    Serial.println("[mag_init] Failed to set BYPASS_EN");
    return false;
  }
  delay(10);

  // Check AK8963 WHO_AM_I (should be 0x48)
  uint8_t who = 0;
  if (!i2c_read_bytes(MAG_ADDR, AK8963_REG_WHO_AM_I, &who, 1)) {
    Serial.println("[mag_init] Failed to read AK8963 WHO_AM_I");
    return false;
  }
  Serial.print("[mag_init] AK8963 WHO_AM_I=0x");
  Serial.println(who, HEX);

  // Power down
  i2c_write_byte(MAG_ADDR, AK8963_REG_CNTL1, 0x00);
  delay(10);

  // Set to 16-bit resolution, continuous measurement mode 2 (100 Hz)
  // 0x16 = 0001 0110b: 16-bit, continuous mode 2
  i2c_write_byte(MAG_ADDR, AK8963_REG_CNTL1, 0x16);
  delay(10);

  Serial.println("[mag_init] Chest magnetometer initialised.");
  return true;
}

// Read AK8963 magnetic field on chest in µT; returns false if no new data
bool mag_read_chest(float &mx_uT, float &my_uT, float &mz_uT) {
  const uint8_t chest_ch = IMU_CHANNEL[0];
  tca_select(chest_ch);

  uint8_t st1 = 0;
  if (!i2c_read_bytes(MAG_ADDR, AK8963_REG_ST1, &st1, 1)) {
    return false;
  }
  if (!(st1 & 0x01)) {
    // no new data
    return false;
  }

  uint8_t buf[7];
  if (!i2c_read_bytes(MAG_ADDR, AK8963_REG_HXL, buf, 7)) {
    return false;
  }

  int16_t raw_x = (int16_t)((buf[1] << 8) | buf[0]); // note: LSB first
  int16_t raw_y = (int16_t)((buf[3] << 8) | buf[2]);
  int16_t raw_z = (int16_t)((buf[5] << 8) | buf[4]);
  uint8_t st2   = buf[6];
  (void)st2; // ignore overflow for now

  mx_uT = raw_x * MAG_SENS;
  my_uT = raw_y * MAG_SENS;
  mz_uT = raw_z * MAG_SENS;

  return true;
}

// ===================== Arduino setup & loop =====================

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("[setup] ESP32-S3 bilateral IMU hub starting...");

  Wire.begin(I2C_SDA, I2C_SCL, I2C_FREQ);

  // Initialize all IMUs (accel+gyro)
  for (int i = 0; i < N_IMU; i++) {
    uint8_t ch = IMU_CHANNEL[i];
    Serial.print("[setup] Initializing IMU ");
    Serial.print(i);
    Serial.print(" (");
    Serial.print(NODE_NAME[i]);
    Serial.print(") on TCA channel ");
    Serial.println(ch);

    if (!imu_init_on_channel(ch)) {
      Serial.print("[setup]   -> FAILED on channel ");
      Serial.println(ch);
    } else {
      Serial.print("[setup]   -> OK on channel ");
      Serial.println(ch);
    }
    delay(50);
  }

  // Initialise magnetometer on chest IMU (index 0)
  if (!mag_init_chest()) {
    Serial.println("[setup] WARNING: chest magnetometer init failed.");
  }

  last_update_us = micros();
}

void loop() {
  uint32_t now_us = micros();
  float dt = (now_us - last_update_us) * 1e-6f;
  if (dt <= 0.0f) dt = 1e-3f;
  last_update_us = now_us;

  const float DEG2RAD = 3.14159265358979f / 180.0f;

  for (int i = 0; i < N_IMU; i++) {
    uint8_t ch = IMU_CHANNEL[i];

    float ax_g, ay_g, az_g;
    float gx_dps, gy_dps, gz_dps;

    bool ok = imu_read_on_channel(ch, ax_g, ay_g, az_g,
                                  gx_dps, gy_dps, gz_dps);
    if (!ok) {
      continue;
    }

    // Gyro to rad/s
    float gx_rad = gx_dps * DEG2RAD;
    float gy_rad = gy_dps * DEG2RAD;
    float gz_rad = gz_dps * DEG2RAD;

    // Update Madgwick filter for this IMU
    filters[i].updateIMU(gx_rad, gy_rad, gz_rad,
                         ax_g, ay_g, az_g,
                         dt);

    float qw, qx, qy, qz;
    filters[i].getQuaternion(qw, qx, qy, qz);

    // Default mag = 0; only chest will override
    float mx_uT = 0.0f, my_uT = 0.0f, mz_uT = 0.0f;
    bool mag_ok = false;
    if (i == 0) {
      mag_ok = mag_read_chest(mx_uT, my_uT, mz_uT);
    }

    // Build NDJSON packet
    uint32_t t_ctrl_us = now_us;        // controller clock
    uint32_t t_node_us = now_us;        // same as controller for wired
    uint32_t seq       = seq_counter[i]++;
    int      node      = NODE_ID[i];
    const char* name   = NODE_NAME[i];
    int      drops     = 0;             // no wireless drops here

    Serial.print("{\"node\":");
    Serial.print(node);
    Serial.print(",\"name\":\"");
    Serial.print(name);
    Serial.print("\",\"t_ctrl_us\":");
    Serial.print(t_ctrl_us);
    Serial.print(",\"t_node_us\":");
    Serial.print(t_node_us);
    Serial.print(",\"seq\":");
    Serial.print(seq);
    Serial.print(",\"drops\":");
    Serial.print(drops);

    // Quaternion
    Serial.print(",\"qw\":"); Serial.print(qw, 6);
    Serial.print(",\"qx\":"); Serial.print(qx, 6);
    Serial.print(",\"qy\":"); Serial.print(qy, 6);
    Serial.print(",\"qz\":"); Serial.print(qz, 6);

    // Raw accel (g)
    Serial.print(",\"ax\":"); Serial.print(ax_g, 6);
    Serial.print(",\"ay\":"); Serial.print(ay_g, 6);
    Serial.print(",\"az\":"); Serial.print(az_g, 6);

    // Raw gyro (deg/s)
    Serial.print(",\"gx\":"); Serial.print(gx_dps, 6);
    Serial.print(",\"gy\":"); Serial.print(gy_dps, 6);
    Serial.print(",\"gz\":"); Serial.print(gz_dps, 6);

    // Magnetometer (µT) — real values for chest, 0 for others
    Serial.print(",\"mx\":"); Serial.print(mx_uT, 6);
    Serial.print(",\"my\":"); Serial.print(my_uT, 6);
    Serial.print(",\"mz\":"); Serial.print(mz_uT, 6);

    Serial.println("}");
  }

  // Optional small delay to avoid spamming at insane rates
  // delay(1);
}
