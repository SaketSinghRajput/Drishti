def test_losses_imports():
    from models import focal_loss_objective
    assert callable(focal_loss_objective)

def test_focal_basic():
    import numpy as np
    from models import focal_loss_objective
    class DummyD:
        def __init__(self, label):
            self._label = label
        def get_label(self):
            return self._label

    preds = np.array([0.0, 1.0, -1.0])
    labels = np.array([0,1,0], dtype=float)
    d = DummyD(labels)
    g,h = focal_loss_objective(preds, d)
    assert g.shape == preds.shape
    assert h.shape == preds.shape
