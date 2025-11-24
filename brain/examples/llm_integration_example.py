"""Example showing usage of LLMIntegration with a minimal config."""
from inference.llm_integration import LLMIntegration


class SimpleConfig:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        parts = key.split('.')
        cur = self._data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


def main():
    cfg = SimpleConfig({
        "inference": {
            "llm_integration": {
                "enabled": False,
                "model_name": "gemini-1.5-flash",
                "timeout_s": 0.5,
            }
        }
    })

    llm = LLMIntegration(cfg)
    features = {"driver_drowsiness_mean": 0.8, "camera_min_distance_m": 2.5}
    explanation = {"message": "high blink rate"}
    sensor_health = {"gps": {"stale": False}}

    print("LLM enabled?", getattr(llm, 'enabled', False))
    print("Calling enhance_risk_assessment (may be gated/off in this example)")
    out = llm.enhance_risk_assessment(features, explanation, sensor_health, 0.75, camera_metadata={"nearest_object_distance_m": 2.5}, window_data={})
    print("Result:", out)


if __name__ == '__main__':
    main()
