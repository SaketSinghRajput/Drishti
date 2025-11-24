import numpy as np
import sys
from unittest import mock
from config.config_loader import ConfigLoader

from inference.explainer import Explainer


def test_explainer_with_mocked_shap():
    cfg = ConfigLoader('config/default_config.yaml')
    # create a fake model with feature names and a _booster attr
    fake_model = mock.Mock()
    fake_model.get_feature_names.return_value = ["a", "b", "c"]
    fake_model._booster = mock.Mock()

    # fake shap module
    fake_shap = mock.Mock()
    fake_tree = mock.Mock()
    # shap_values returns array matching features
    fake_tree.shap_values.return_value = np.array([[0.2, -0.1, 0.05]])
    fake_shap.TreeExplainer.return_value = fake_tree

    with mock.patch.dict('sys.modules', {'shap': fake_shap}):
        expl = Explainer(cfg, model=fake_model)
        out = expl.explain({"a": 1.0, "b": 2.0, "c": 3.0}, 0.5)
        assert out.get('explainability_available') is True
        assert 'message' in out
        assert isinstance(out.get('top_features'), list)
