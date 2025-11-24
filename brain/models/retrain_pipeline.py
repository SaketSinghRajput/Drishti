import json
import logging
import os
from typing import Optional, Dict, Any, List

import pandas as pd

from utils.data_validator import DataValidator
from models.model_registry import ModelRegistry
from models.ab_testing import ABTestingFramework
from models.regression_detector import RegressionDetector
from models.model_deployer import ModelDeployer
from models.retraining_scheduler import RetrainingScheduler

from config.config_loader import ConfigLoader
try:
    from models.train_xgboost import XGBoostTrainer
except Exception:
    XGBoostTrainer = None
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        retr_cfg = cfg.get('retraining', {})
        self.config = cfg
        self.validator = DataValidator(retr_cfg.get('validation', {}))
        self.registry = ModelRegistry(retr_cfg.get('registry', {}).get('db_path', 'data/model_registry.db'))
        self.abtest = ABTestingFramework(retr_cfg.get('ab_testing', {}))
        self.regdet = RegressionDetector(retr_cfg.get('regression_detection', {}))
        # pass registry into deployer config for deployment recording
        dep_cfg = dict(retr_cfg.get('deployment', {}))
        dep_cfg['_registry'] = self.registry
        self.deployer = ModelDeployer(dep_cfg)
        self.scheduler = RetrainingScheduler(retr_cfg.get('scheduler', {}))

    def run_full_pipeline(self, dataset_csv: str, output_dir: str, incremental: bool = True) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        report = {'stages': {}}
        # stage 1: load dataset
        try:
            # dataset_csv may be a CSV file path or a labeled dataset directory
            if os.path.isdir(dataset_csv):
                df = None
                report['stages']['load'] = {'ok': True, 'rows': None, 'note': 'directory'}
            else:
                df = pd.read_csv(dataset_csv)
                report['stages']['load'] = {'ok': True, 'rows': len(df)}
        except Exception as e:
            report['stages']['load'] = {'ok': False, 'error': str(e)}
            return report

        # validate
        val = self.validator.generate_validation_report(df) if df is not None else {'passed': True}
        report['stages']['validation'] = val
        if not val.get('passed'):
            logger.warning('Validation failed, aborting pipeline')
            return report
        # Train: integrate with project's XGBoost trainer if available
        cfg_loader = ConfigLoader()
        # if pipeline provided an in-memory config dict, inject it into loader for trainer access
        if isinstance(self.config, dict):
            try:
                cfg_loader._data = self.config
            except Exception:
                pass

        horizons = self.config.get('labeling', {}).get('prediction_horizons') or [self.config.get('model', {}).get('prediction_horizon_s', 10)]
        trained_models: List[Dict[str, Any]] = []
        for h in horizons:
            try:
                if XGBoostTrainer is None:
                    raise RuntimeError('Trainer not available')
                trainer = XGBoostTrainer(cfg_loader, horizon=h)
                # If dataset is a directory, let trainer prepare data itself
                if os.path.isdir(dataset_csv):
                    X_train, X_val, X_test, y_train, y_val, y_test, groups_train, sample_weights = trainer.load_and_prepare_data(dataset_csv)
                else:
                    # dataset is a CSV: derive X/y and split
                    if df is None:
                        raise RuntimeError('No dataset available')
                    label_candidates = [f'is_accident_h{h}', 'label', 'is_accident']
                    label_col = next((c for c in label_candidates if c in df.columns), None)
                    if label_col is None:
                        raise RuntimeError(f'Label column not found for horizon {h}')
                    X = df.drop(columns=[label_col])
                    y = df[label_col]
                    test_size = float(cfg_loader.get('training.test_size', 0.2))
                    val_size = float(cfg_loader.get('training.val_size', 0.1))
                    # first split train+val and test
                    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=int(cfg_loader.get('training.random_seed', 42)))
                    # then split train and val
                    val_rel = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
                    if val_rel > 0:
                        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_rel, random_state=int(cfg_loader.get('training.random_seed', 42)))
                    else:
                        X_train, y_train = X_train_val, y_train_val
                        X_val, y_val = None, None
                    groups_train = None
                    sample_weights = None

                # train
                train_meta = trainer.train(X_train, y_train, X_val=X_val, y_val=y_val, groups=groups_train, sample_weights=sample_weights)
                # save artifact
                model_path = os.path.join(output_dir, f'model_h{h}.xgb')
                trainer.save_model(model_path)
                # try ONNX export
                onnx_path = os.path.join(output_dir, f'model_h{h}.onnx')
                try:
                    trainer.export_onnx(onnx_path)
                except Exception:
                    onnx_path = None
                # evaluate if possible
                metrics = {}
                try:
                    if 'X_test' in locals() or (locals().get('X_test') is not None):
                        if X_test is not None and y_test is not None:
                            eval_out = trainer.evaluate(X_test, y_test, os.path.join(output_dir, f'eval_h{h}'))
                            metrics = eval_out.get('metrics', {}) if isinstance(eval_out, dict) else {}
                except Exception:
                    metrics = {}

                model_id = self.registry.register_model(horizon=h, model_path=model_path, onnx_path=onnx_path, config_snapshot=self.config, dataset_hash=val.get('dataset_hash', ''), metrics=metrics)
                trained_models.append({'horizon': h, 'model_id': model_id, 'model_path': model_path, 'onnx_path': onnx_path, 'metrics': metrics})
                report['stages'].setdefault('training', []).append({'horizon': h, 'model_id': model_id, 'metrics': metrics})
            except Exception as exc:
                report['stages'].setdefault('training', []).append({'horizon': h, 'ok': False, 'error': str(exc)})

        # AB test and regression detection: for each trained model compare with baseline deployed model of same horizon
        report['stages']['ab_test'] = {}
        for tm in trained_models:
            h = tm['horizon']
            model_id = tm['model_id']
            baseline = self.registry.get_latest_model(horizon=h, status='deployed')
            if not baseline:
                report['stages']['ab_test'][h] = {'skipped': True, 'reason': 'no_baseline_deployed'}
                continue
            # prepare hold-out X_test/y_test
            X_test_local = None
            y_test_local = None
            try:
                if os.path.isdir(dataset_csv):
                    # cannot easily prepare hold-out here; skip if trainer provided dataset loader
                    report['stages']['ab_test'][h] = {'skipped': True, 'reason': 'holdout_unavailable_for_dir'}
                    continue
                else:
                    # reuse df split performed above if available
                    label_candidates = [f'is_accident_h{h}', 'label', 'is_accident']
                    label_col = next((c for c in label_candidates if c in df.columns), None)
                    if label_col is None:
                        report['stages']['ab_test'][h] = {'skipped': True, 'reason': 'no_label_col'}
                        continue
                    X_all = df.drop(columns=[label_col])
                    y_all = df[label_col]
                    # create a holdout split
                    _, X_test_local, _, y_test_local = train_test_split(X_all, y_all, test_size=float(cfg_loader.get('training.test_size', 0.2)), random_state=int(cfg_loader.get('training.random_seed', 42)))
            except Exception:
                report['stages']['ab_test'][h] = {'skipped': True, 'reason': 'holdout_prep_failed'}
                continue

            try:
                # load new and baseline model into trainers for prediction
                new_tr = XGBoostTrainer(cfg_loader, horizon=h)
                new_tr.load_model(tm['model_path'])
                base_tr = XGBoostTrainer(cfg_loader, horizon=h)
                base_tr.load_model(baseline.get('model_path'))

                def pred_new(X):
                    return new_tr.predict_proba(X)

                def pred_base(X):
                    return base_tr.predict_proba(X)

                cmp = self.abtest.compare_models(pred_base, pred_new, X_test_local.values, y_test_local.values)
                report['stages']['ab_test'][h] = {'ok': True, 'report': cmp}
            except Exception as exc:
                report['stages']['ab_test'][h] = {'ok': False, 'error': str(exc)}

        # regression detection per-horizon using registry or recomputed baseline metrics
        report['stages']['regression'] = {}
        for tm in trained_models:
            h = tm['horizon']
            model_id = tm['model_id']
            baseline = self.registry.get_latest_model(horizon=h, status='deployed')
            baseline_metrics = {}
            if baseline:
                try:
                    baseline_metrics = self.registry.get_model_metrics(baseline['id'])
                except Exception:
                    baseline_metrics = {}
            new_metrics = tm.get('metrics', {}) or {}
            reg = self.regdet.detect_regression(new_metrics, baseline_metrics)
            report['stages']['regression'][h] = reg
            if reg.get('has_regression') and reg.get('severity') == 'critical' and self.config.get('retraining', {}).get('regression_detection', {}).get('abort_on_critical_regression', True):
                report['stages']['deploy'] = {'skipped': True, 'reason': 'critical_regression', 'horizon': h}
                return report

        # package
        # package trained artifacts (handle multiple horizons; package per-horizon)
        report['stages']['package'] = {}
        for tm in trained_models:
            model_meta = {'version': self.registry.get_model_by_id(tm['model_id']).get('version'), 'horizon': tm['horizon'], 'created_at': ''}
            package = self.deployer.package_model(model_meta, tm['model_path'], tm.get('onnx_path'))
            report['stages']['package'][tm['horizon']] = {'ok': True, 'package_path': package, 'model_id': tm['model_id']}

        # deploy if enabled
        # deploy if enabled
        if self.config.get('retraining', {}).get('deployment', {}).get('auto_deploy', False):
            # deploy each horizon package
            deploy_results = {}
            for h, pinfo in report['stages']['package'].items():
                pkg = pinfo.get('package_path')
                mid = pinfo.get('model_id')
                res = self.deployer.deploy_to_all_devices(pkg, model_id=mid)
                deploy_results[h] = res
                try:
                    self.registry.update_model_status(mid, 'deployed')
                except Exception:
                    pass
            report['stages']['deploy'] = {'ok': True, 'results': deploy_results}
        else:
            report['stages']['deploy'] = {'ok': False, 'reason': 'auto_deploy_disabled'}

        # save summary
        out = os.path.join(output_dir, 'retraining_report.json')
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        report['report_path'] = out
        return report

    def run_validation_only(self, dataset_csv: str) -> Dict[str, Any]:
        import pandas as pd
        df = pd.read_csv(dataset_csv)
        return self.validator.generate_validation_report(df)

    def run_deployment_only(self, model_id: int) -> Dict[str, Any]:
        # look up model by id and package/deploy
        m = self.registry.get_model_by_id(model_id)
        if not m:
            return {'ok': False, 'reason': 'model_not_found'}
        model_meta = {'version': m.get('version'), 'horizon': m.get('horizon'), 'created_at': m.get('created_at')}
        package = self.deployer.package_model(model_meta, m.get('model_path'), m.get('onnx_path'))
        results = self.deployer.deploy_to_all_devices(package, model_id=model_id)
        # record status
        try:
            self.registry.update_model_status(model_id, 'deployed')
        except Exception:
            pass
        return {'ok': True, 'package': package, 'results': results}
