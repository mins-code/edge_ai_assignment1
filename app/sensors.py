"""Service envelope for a powered single-cell doorbell reporting an associated Wi-Fi link."""

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
BOUNDS = {
    "battery_v": (2.0, 4.5),
    "temp_c": (-40.0, 85.0),
    "rings_per_day": (0.0, 86400.0),
    "wifi_rssi_dbm": (-120.0, 0.0),  # Zero is the missing-link sentinel, exclusive.
    "age_months": (0.0, 240.0),
}
