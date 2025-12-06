#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
#include <ArduinoJson.h>
#include <time.h>

// Uncomment for raw hex + JSON debug output
// #define DEBUG  // Uncomment for raw hex + JSON debug output
#define DEBUG_BUFFER_SIZE 256

// Configuration constants (from plan)
#define SAFETY_MANUFACTURER_ID 0xFFFF
#define PACKET_SERVICE_UUID "0000FFE0-0000-1000-8000-00805F9B34FB"
#define CHAR_UUID "0000FFE1-0000-1000-8000-00805F9B34FB"
#define TS_TOLERANCE_MS 10000UL
#define BUFFER_SIZE 100
#define SCAN_INTERVAL_S 1 // Scan for only 1 second to allow incoming connections
#define SUMMARY_INTERVAL_MS 5000UL

// BLE security mode selector (compile-time):
// 0 = BLESecurity disabled (open access, rely on library defaults)
// 1 = Minimal mode: setAuthenticationMode(ESP_LE_AUTH_NO_BOND), no init encryption keys
// 2 = Original mode: setAuthenticationMode(ESP_LE_AUTH_NO_BOND) and setInitEncryptionKey(ENC|ID)
#ifndef BLE_SECURITY_MODE
#define BLE_SECURITY_MODE 1
#endif

struct SafetyPacket {
  String vid;
  int fatigue_pct;
  float risk_pct;
  String battery_severity;
  float max_cell_temp;
  int ble_neighbors;
  float lat;
  float lon;
  float speed_ms;
  unsigned long ts;
};

BLEScan* pBLEScan;
unsigned long last_summary = 0;
String vids[BUFFER_SIZE];
int vid_idx = 0;
int vid_count = 0;
unsigned long rx_times[BUFFER_SIZE];
unsigned long total_packets = 0;
// GATT server globals
BLEServer* pServer = nullptr;
BLECharacteristic* pCharacteristic = nullptr;

// Connection callback to log connect/disconnect and restart advertising
class ServerConnectionCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    Serial.println("[GATT] Client connected");
  }
  void onDisconnect(BLEServer* server) override {
    Serial.println("[GATT] Client disconnected");
    // restart advertising so the device remains discoverable
    BLEDevice::getAdvertising()->start();
    Serial.println("[GATT] Advertising restarted");
  }
};

// Forward declarations
void parseManufacturerData(BLEAdvertisedDevice device);
void parseServiceData(BLEAdvertisedDevice device);
SafetyPacket parseAndValidatePacket(const String &payload, int rssi);
bool validatePacket(const SafetyPacket &packet);
void printPacket(const SafetyPacket &packet, int rssi);
void addVid(const String &vid);
void printSummary();

// Characteristic callback to handle incoming GATT writes (file-scope for reuse)
class SafetyPacketCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pChar) override {
    String value = pChar->getValue();
    if (value.length() == 0) {
      Serial.println("[GATT] Empty write received");
      return;
    }
    // Construct Arduino String using explicit length to avoid truncation
    String payload = String(value.c_str(), (int)value.length());
    Serial.printf("[GATT] Received write (%u bytes)\n", (unsigned)value.length());
    SafetyPacket pkt = parseAndValidatePacket(payload, 0);
    if (pkt.vid.length() == 0) {
      Serial.println("[GATT] Received invalid packet or missing VID");
      return;
    }
    printPacket(pkt, 0);
    addVid(pkt.vid);
    total_packets++;
  }
};

// Debug helpers (only active when DEBUG is defined)
void printHex(const uint8_t* data, size_t len, const char* label);
void printParsedJson(const SafetyPacket& pkt);

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("[ESP32] EV Safety BLE Receiver Started");

  // Configure the BLE device local name. We run dual-mode: a scanner (central)
  // that observes advertisements, and a GATT server (peripheral) that accepts
  // JSON writes from a Raspberry Pi sender. Advertise as "node-1" so external
  // scanners (Bleak, nRF Connect) can discover and connect.
  BLEDevice::init("node-1");
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setActiveScan(true);
  pBLEScan->setInterval(100);
  pBLEScan->setWindow(99);

  // --- GATT Server setup -------------------------------------------------
  // Create a BLE server and a service with a writable characteristic. When
  // a remote client writes JSON to the characteristic, the callback will
  // parse and display the packet using existing helpers.
  pServer = BLEDevice::createServer();
  // Register connection callbacks that log events and restart advertising
  pServer->setCallbacks(new ServerConnectionCallbacks());

  // Configure BLE security. Use compile-time mode to allow experimenting
  // with reduced/disabled security to avoid pairing prompts on some hosts.
  // Modes:
  // 0 - BLESecurity disabled (open access)
  // 1 - Minimal: no bonding, no init encryption keys
  // 2 - Full (original): no bonding, init ENC|ID keys
#if BLE_SECURITY_MODE == 0
  Serial.println("[GATT] Security: BLESecurity disabled (open access)");
#else
  BLESecurity* pSecurity = new BLESecurity();
  pSecurity->setAuthenticationMode(ESP_LE_AUTH_NO_BOND);
  pSecurity->setCapability(ESP_IO_CAP_NONE);
#if BLE_SECURITY_MODE == 1
  // Minimal: avoid requesting init encryption keys to reduce pairing-like behavior
  pSecurity->setInitEncryptionKey(0);
  Serial.println("[GATT] Security: No bonding, minimal encryption mask (0)");
#else
  // Original behavior: set ENC | ID masks (may cause encryption negotiation)
  pSecurity->setInitEncryptionKey(ESP_BLE_ENC_KEY_MASK | ESP_BLE_ID_KEY_MASK);
  Serial.println("[GATT] Security: No bonding, encryption keys set (ENC|ID)");
#endif
#endif
  BLEService* pService = pServer->createService(PACKET_SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
      CHAR_UUID,
      BLECharacteristic::PROPERTY_WRITE
  );

  // Set the characteristic callback (class defined at file scope)
  pCharacteristic->setCallbacks(new SafetyPacketCallbacks());
  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(PACKET_SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // recommended minimum connection interval
  pAdvertising->start();
  Serial.println("[GATT] Server started, advertising as 'node-1'");
  // -----------------------------------------------------------------------

  last_summary = millis();
}

void loop() {
  // Start blocking scan for SCAN_INTERVAL_S seconds
  BLEScanResults* foundDevices = pBLEScan->start(SCAN_INTERVAL_S, false);
  int count = foundDevices ? foundDevices->getCount() : 0;
  // Log number of devices found in this scan interval
  Serial.printf("[SCAN] Duration: %d s, Devices found: %d\n", SCAN_INTERVAL_S, count);
  for (int i = 0; i < count; i++) {
    BLEAdvertisedDevice device = foundDevices->getDevice(i);
    if (device.haveManufacturerData()) {
      parseManufacturerData(device);
    }
    if (device.haveServiceData()) {
      parseServiceData(device);
    }
  }
  pBLEScan->clearResults();

  // Allow some breathing room for advertising / incoming connections
  // when running dual-mode (scanner + GATT server). A short delay helps
  // the ESP32 accept and process incoming GATT connections from hosts.
  delay(2000); // Wait 2 seconds between scans

  if (millis() - last_summary >= SUMMARY_INTERVAL_MS) {
    printSummary();
  }
}

void parseManufacturerData(BLEAdvertisedDevice device) {
  String md = device.getManufacturerData();
  // need at least 2 bytes header (manufacturer id). payload may be zero-length.
  if (md.length() < 2) return;

  // parse little-endian manufacturer id
  uint16_t man_id = ((uint8_t)md[1] << 8) | (uint8_t)md[0];
  if (man_id != SAFETY_MANUFACTURER_ID) return;

  // payload starts at offset 2 per plan. Construct String with explicit length
  // to avoid C-string truncation if payload contains NULs or non-UTF bytes.
  int payload_len = (md.length() > 2) ? (int)(md.length() - 2) : 0;
  String payload;
  if (payload_len > 0) {
    const char* start = md.c_str() + 2;
    payload = String(start, payload_len);
  } else {
    payload = String("");
  }
  #ifdef DEBUG
  printHex((const uint8_t*)md.c_str(), md.length(), "Manufacturer Data");
  #endif

  SafetyPacket pkt = parseAndValidatePacket(payload, device.getRSSI());
  if (pkt.vid.length() == 0) return;

  #ifdef DEBUG
  printParsedJson(pkt);
  #endif

  printPacket(pkt, device.getRSSI());
  addVid(pkt.vid);
  total_packets++;
}

void parseServiceData(BLEAdvertisedDevice device) {
  if (!device.haveServiceData()) return;
  String svc = device.getServiceData();
  BLEUUID svcUUID = device.getServiceDataUUID();
  String uuidStr = svcUUID.toString().c_str();
  if (uuidStr != String(PACKET_SERVICE_UUID)) return;

  // Use explicit-length construction to avoid truncation
  String payload = String(svc.c_str(), (int)svc.length());
  #ifdef DEBUG
  printHex((const uint8_t*)svc.c_str(), svc.length(), "Service Data");
  #endif

  SafetyPacket pkt = parseAndValidatePacket(payload, device.getRSSI());
  if (pkt.vid.length() == 0) return;

  #ifdef DEBUG
  printParsedJson(pkt);
  #endif

  printPacket(pkt, device.getRSSI());
  addVid(pkt.vid);
  total_packets++;
}

SafetyPacket parseAndValidatePacket(const String &payload, int rssi) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.print("[ERROR] JSON parse failed: ");
    Serial.println(err.c_str());
    return SafetyPacket();
  }

  SafetyPacket pkt;
  // Support multiple aliases for the vehicle identifier: full key, compact key, and sender's vehicle_hash
  // Accepts: "vid", "v", or "vehicle_hash"
  pkt.vid = String((const char*)(doc["vid"] | doc["v"] | doc["vehicle_hash"] | ""));
  pkt.fatigue_pct = doc["fatigue"] | doc["f"] | 0;
  pkt.risk_pct = (float)(doc["risk"] | doc["r"] | 0.0);
  pkt.battery_severity = String((const char*)(doc["battery_severity"] | "unknown"));
  pkt.max_cell_temp = (float)(doc["max_cell_temp"] | 0.0);
  pkt.ble_neighbors = doc["ble_neighbors"] | 0;
  pkt.lat = (float)(doc["lat"] | 0.0);
  pkt.lon = (float)(doc["lon"] | 0.0);
  pkt.speed_ms = (float)(doc["speed"] | 0.0);
  // ArduinoJson returns long long for large integers; cast carefully
  unsigned long ts_val = 0UL;
  if (doc.containsKey("ts")) {
    ts_val = (unsigned long)doc["ts"];
  }
  pkt.ts = ts_val;

  if (!validatePacket(pkt)) return SafetyPacket();

  return pkt;
}

bool validatePacket(const SafetyPacket &packet) {
  if (packet.vid.length() == 0) {
    Serial.println("[ERROR] Missing VID");
    return false;
  }
  unsigned long now_ms = millis();
  // Per plan, compare ts to millis() with tolerance (note: plan specified this)
  if (packet.ts != 0 && packet.ts < now_ms - TS_TOLERANCE_MS) {
    unsigned long age = now_ms - packet.ts;
    Serial.print("[WARN] Stale packet: ts=");
    Serial.print(packet.ts);
    Serial.print(" age=");
    Serial.print(age);
    Serial.println("ms");
    return false;
  }
  return true;
}

void printPacket(const SafetyPacket &packet, int rssi) {
  char time_str[64] = {0};
  time_t now = time(nullptr);
  struct tm *tm_info = localtime(&now);
  if (tm_info) {
    strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S%z", tm_info);
  } else {
    strcpy(time_str, "1970-01-01T00:00:00+0000");
  }

  String risk_level = "LOW";
  if (packet.risk_pct > 80.0) risk_level = "CRITICAL";
  else if (packet.risk_pct > 60.0) risk_level = "HIGH";
  else if (packet.risk_pct > 33.0) risk_level = "MEDIUM";

  // Per plan: approximate RX latency
  unsigned long latency_ms = 0;
  if (packet.ts != 0) latency_ms = millis() - (packet.ts % 1000000UL);

  String vid_short = packet.vid;
  if (vid_short.length() > 8) vid_short = vid_short.substring(0, 8);

  Serial.printf("[EV-PACKET] %s VID: %s FATIGUE: %d%% RISK: %.1f%% (%s) BATTERY: max_temp=%.1fC severity=%s BLE_NEIGHBORS: %d LOCATION: %.6f, %.6f speed=%.2f m/s RX_LATENCY: %lu ms TS_VALID: yes\n",
                time_str,
                vid_short.c_str(),
                packet.fatigue_pct,
                packet.risk_pct,
                risk_level.c_str(),
                packet.max_cell_temp,
                packet.battery_severity.c_str(),
                packet.ble_neighbors,
                packet.lat,
                packet.lon,
                packet.speed_ms,
                latency_ms);
}

#ifdef DEBUG
void printHex(const uint8_t* data, size_t len, const char* label) {
  Serial.printf("[DEBUG-HEX] %s: %u bytes\n", label, (unsigned)len);
  const size_t per_line = 16;
  for (size_t i = 0; i < len; i += per_line) {
    // hex bytes
    Serial.print("  ");
    for (size_t j = 0; j < per_line; j++) {
      size_t idx = i + j;
      if (idx < len) {
        Serial.printf("0x%02X ", data[idx]);
      } else {
        Serial.print("     ");
      }
    }
    // ASCII preview
    Serial.print("  ");
    for (size_t j = 0; j < per_line; j++) {
      size_t idx = i + j;
      if (idx < len) {
        char c = (char)data[idx];
        if (c >= 32 && c <= 126) Serial.print(c);
        else Serial.print('.');
      }
    }
    Serial.println();
  }
}

void printParsedJson(const SafetyPacket& pkt) {
  Serial.print("[DEBUG-JSON] {");
  Serial.printf("vid:\"%s\", ", pkt.vid.c_str());
  Serial.printf("fatigue:%d, ", pkt.fatigue_pct);
  Serial.printf("risk:%.1f, ", pkt.risk_pct);
  Serial.printf("battery_severity:\"%s\", ", pkt.battery_severity.c_str());
  Serial.printf("max_cell_temp:%.1f, ", pkt.max_cell_temp);
  Serial.printf("ble_neighbors:%d, ", pkt.ble_neighbors);
  Serial.printf("lat:%.6f, lon:%.6f, ", pkt.lat, pkt.lon);
  Serial.printf("speed:%.2f, ts:%lu", pkt.speed_ms, pkt.ts);
  Serial.println(" }");
}
#endif

void addVid(const String &vid) {
  if (vid.length() == 0) return;
  // check if present
  for (int i = 0; i < vid_count; i++) {
    int idx = (vid_idx + BUFFER_SIZE - vid_count + i) % BUFFER_SIZE;
    if (vids[idx] == vid) {
      rx_times[idx] = millis();
      return;
    }
  }
  // not found, add at vid_idx
  vids[vid_idx] = vid;
  rx_times[vid_idx] = millis();
  vid_idx = (vid_idx + 1) % BUFFER_SIZE;
  if (vid_count < BUFFER_SIZE) vid_count++;
}

void printSummary() {
  unsigned long now = millis();
  int recent_count = 0;
  for (int i = 0; i < vid_count; i++) {
    int idx = (vid_idx + BUFFER_SIZE - vid_count + i) % BUFFER_SIZE;
    if (now - rx_times[idx] <= 30000UL) recent_count++;
  }
  Serial.printf("[SUMMARY] Unique VIDs last 30s: %d | Total packets: %lu\n", recent_count, total_packets);
  last_summary = now;
}
