/*
  controller_side.ino — ESP32 controller / receiver (NDJSON + commands)
  ---------------------------------------------------------------------
  • Receives ESP-NOW IMU packets from multiple nodes
  • Streams NDJSON over Serial for Python
  • Sends broadcast commands: CALIBRATE / START_STREAM / STOP_STREAM

  Packet from node (must match node_side.ino):
    #pragma pack(push, 1)
    struct ImuPacket {
      uint8_t  magic;      // 0xA5
      uint8_t  version;    // 1
      uint8_t  nodeId;
      uint8_t  reserved;

      uint32_t seq;
      uint32_t t_node_us;

      float    q[4];       // w,x,y,z
      float    acc[3];     // accel in g
      float    gyr[3];     // gyro in deg/s

      uint8_t  flags;      // bit0=mag_used, bit1=imu_ok
    };
    #pragma pack(pop)
*/

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

// ========== CONFIG ==========
#define WIFI_CHANNEL           1
#define SERIAL_BAUD            115200
#define TIME_BEACON_MS         500     // send a small time beacon over Serial every 500 ms

// Broadcast peer (for commands from controller to all nodes)
static const uint8_t BROADCAST_ADDR[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

// Optional: map IDs to human names
static const char* nameForId(uint8_t id) {
  switch (id) {
    case 1: return "wrist";
    case 2: return "elbow";
    case 3: return "shoulder";
    case 4: return "chest";
    default: return "node";
  }
}

// ========== Command struct & enums (must match node) ==========
#pragma pack(push, 1)
struct CtrlCommand {
  uint8_t  magic;    // 0xC3
  uint8_t  version;  // 1
  uint8_t  cmd;      // Cmd code
  uint8_t  reserved;
  uint32_t arg;      // optional; not used now
};
#pragma pack(pop)

enum CtrlCmd : uint8_t {
  CMD_CALIBRATE    = 1,
  CMD_START_STREAM = 2,
  CMD_STOP_STREAM  = 3,
};

// ========== IMU Packet (must match node) ==========
#pragma pack(push, 1)
typedef struct {
  uint8_t  magic;      // 0xA5
  uint8_t  version;    // 1
  uint8_t  nodeId;
  uint8_t  reserved;

  uint32_t seq;
  uint32_t t_node_us;

  float    q[4];       // w,x,y,z
  float    acc[3];     // in g
  float    gyr[3];     // in deg/s

  uint8_t  flags;      // bit0=mag_used, bit1=imu_ok
} ImuPacket;
#pragma pack(pop)

// Simple stats per node
struct NodeStats {
  uint32_t lastSeq  = 0;
  uint32_t drops    = 0;
  uint32_t lastRxUs = 0;
  bool     seen     = false;
};

static NodeStats statsById[256];   // indexed by nodeId

// Forward declarations
static void onRecv(const esp_now_recv_info *info, const uint8_t *incomingData, int len);
static void maybeSendTimeBeacon();
void sendCommandToAll(uint8_t cmd);

// ========== ESP-NOW receive callback (new core signature) ==========
static void onRecv(const esp_now_recv_info *info,
                   const uint8_t *incomingData,
                   int len) {
  if (len < (int)sizeof(ImuPacket)) {
    // not our packet type
    return;
  }

  ImuPacket p;
  memcpy(&p, incomingData, sizeof(ImuPacket));

  if (p.magic != 0xA5) return;
  if (p.version != 1)  return;

  uint8_t id = p.nodeId;
  uint32_t rxUs = micros();

  NodeStats &st = statsById[id];

  if (st.seen) {
    uint32_t expected = st.lastSeq + 1;
    if (p.seq != expected && p.seq > st.lastSeq) {
      uint32_t missed = p.seq - expected;
      st.drops += missed;
    }
  }
  st.seen    = true;
  st.lastSeq = p.seq;
  st.lastRxUs = rxUs;

  // NDJSON out
  Serial.print('{');

  Serial.print("\"t_ctrl_us\":");
  Serial.print((uint32_t)rxUs);

  Serial.print(",\"node\":");
  Serial.print((int)id);

  Serial.print(",\"name\":\"");
  Serial.print(nameForId(id));
  Serial.print('"');

  Serial.print(",\"seq\":");
  Serial.print(p.seq);

  Serial.print(",\"drops\":");
  Serial.print(st.drops);

  Serial.print(",\"t_node_us\":");
  Serial.print(p.t_node_us);

  // Quaternion
  Serial.print(",\"qw\":");
  Serial.print(p.q[0], 6);
  Serial.print(",\"qx\":");
  Serial.print(p.q[1], 6);
  Serial.print(",\"qy\":");
  Serial.print(p.q[2], 6);
  Serial.print(",\"qz\":");
  Serial.print(p.q[3], 6);

  // Accel (g)
  Serial.print(",\"ax\":");
  Serial.print(p.acc[0], 6);
  Serial.print(",\"ay\":");
  Serial.print(p.acc[1], 6);
  Serial.print(",\"az\":");
  Serial.print(p.acc[2], 6);

  // Gyro (deg/s)
  Serial.print(",\"gx\":");
  Serial.print(p.gyr[0], 6);
  Serial.print(",\"gy\":");
  Serial.print(p.gyr[1], 6);
  Serial.print(",\"gz\":");
  Serial.print(p.gyr[2], 6);

  // Flags
  Serial.print(",\"mag_used\":");
  Serial.print((p.flags & 0x01) ? 1 : 0);

  Serial.print(",\"imu_ok\":");
  Serial.print((p.flags & 0x02) ? 1 : 0);

  Serial.println('}');
}

// ========== Time beacon over Serial ==========
static uint32_t lastBeaconMs = 0;

static void maybeSendTimeBeacon() {
  uint32_t nowMs = millis();
  if (nowMs - lastBeaconMs < TIME_BEACON_MS) return;
  lastBeaconMs = nowMs;

  Serial.print('{');
  Serial.print("\"type\":\"beacon\",\"t_ctrl_us\":");
  Serial.print((uint32_t)micros());
  Serial.println('}');
}

// ========== Send broadcast command to all nodes ==========
void sendCommandToAll(uint8_t cmd) {
  CtrlCommand c{};
  c.magic   = 0xC3;
  c.version = 1;
  c.cmd     = cmd;
  c.reserved= 0;
  c.arg     = 0;

  esp_err_t e = esp_now_send(BROADCAST_ADDR, (uint8_t*)&c, sizeof(c));
  if (e != ESP_OK) {
    Serial.printf("[CMD] send err=%d\n", e);
  } else {
    Serial.printf("[CMD] sent %u\n", (unsigned)cmd);
  }
}

// ========== SETUP / LOOP ==========
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  Serial.println("\n[Controller] boot NDJSON + command mode");

  // WiFi STA & fixed channel
  WiFi.mode(WIFI_STA);
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  // ESP-NOW init
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] init failed");
    while (true) {
      delay(1000);
    }
  }

  // RX callback (note: new signature)
  esp_now_register_recv_cb(onRecv);

  // Add broadcast peer (for commands)
  esp_now_peer_info_t peer{};
  memset(&peer, 0, sizeof(peer));
  memcpy(peer.peer_addr, BROADCAST_ADDR, 6);
  peer.channel = WIFI_CHANNEL;
  peer.encrypt = false;

  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("[ESP-NOW] add broadcast peer failed");
  }

  Serial.println("[Controller] ready: 'c'=CALIB, 's'=START, 'x'=STOP");
}

void loop() {
  maybeSendTimeBeacon();

  // Keyboard commands over Serial
  if (Serial.available()) {
    char ch = Serial.read();
    if (ch == 'c' || ch == 'C') {
      Serial.println("[Host] CALIBRATE all nodes");
      sendCommandToAll(CMD_CALIBRATE);
    } else if (ch == 's' || ch == 'S') {
      Serial.println("[Host] START_STREAM all nodes");
      sendCommandToAll(CMD_START_STREAM);
    } else if (ch == 'x' || ch == 'X') {
      Serial.println("[Host] STOP_STREAM all nodes");
      sendCommandToAll(CMD_STOP_STREAM);
    }
  }
}
