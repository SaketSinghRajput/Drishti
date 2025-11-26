from brain.data_collection.camera_adapter_intent import compute_pedestrian_intent_features, compute_cyclist_features


def make_person_with_intent(score):
    return {'objects': [{'pose_data': {'pedestrian_intent_score': score}}]}


def make_cyclist_with_angle(angle):
    return {'objects': [{'class': 'bicycle', 'pose_data': {'cyclist_lean_angle': angle}}]}


def test_pedestrian_intent_score():
    samples = [make_person_with_intent(0.8), make_person_with_intent(0.9), make_person_with_intent(0.2)]
    feats = compute_pedestrian_intent_features(samples)
    assert feats['pedestrian_intent_score_max'] >= 0.9
    assert feats['pedestrians_walking_towards_count'] >= 1


def test_cyclist_lean_detection():
    samples = [make_cyclist_with_angle(20.0), make_cyclist_with_angle(10.0)]
    feats = compute_cyclist_features(samples)
    assert feats['avg_cyclist_lean_angle'] > 0
    assert feats['cyclists_maneuvering_count'] >= 1

def test_no_pose_data():
    samples = []
    assert compute_pedestrian_intent_features(samples) == {}
    assert compute_cyclist_features(samples) == {}
