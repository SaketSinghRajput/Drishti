import os
import tarfile
import json
import subprocess
import time
from typing import Dict, Any, List, Optional


class ModelDeployer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.package_dir = cfg.get('package_dir', 'data/model_packages')
        self.edge_devices = cfg.get('edge_devices', [])
        self.method = cfg.get('deployment_method', 'scp')
        self.api_endpoint = cfg.get('api_endpoint')
        self.api_key_env = cfg.get('api_key_env')
        os.makedirs(self.package_dir, exist_ok=True)
        # optional registry instance to record deployments
        self.registry = cfg.get('_registry')

    def package_model(self, model_meta: Dict[str, Any], model_path: str, onnx_path: Optional[str] = None) -> str:
        version = model_meta.get('version', 'v0')
        horizon = model_meta.get('horizon', 'h')
        name = f"model_{version}_h{horizon}.tar.gz"
        out = os.path.join(self.package_dir, name)
        with tarfile.open(out, 'w:gz') as tf:
            tf.add(model_path, arcname=os.path.basename(model_path))
            if onnx_path and os.path.exists(onnx_path):
                tf.add(onnx_path, arcname=os.path.basename(onnx_path))
            meta = {'meta': model_meta}
            meta_path = os.path.join(self.package_dir, f"{version}_meta.json")
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
            tf.add(meta_path, arcname=os.path.basename(meta_path))
        return out

    def deploy_to_edge(self, package_path: str, device: Dict[str, Any]) -> Dict[str, Any]:
        host = device.get('host')
        user = device.get('user')
        path = device.get('path')
        if self.method == 'scp':
            cmd = ['scp', package_path, f"{user}@{host}:{path}"]
            try:
                subprocess.check_call(cmd)
                return {'device': device.get('id'), 'ok': True}
            except Exception as e:
                return {'device': device.get('id'), 'ok': False, 'error': str(e)}
        elif self.method == 'rsync':
            cmd = ['rsync', '-avz', package_path, f"{user}@{host}:{path}"]
            try:
                subprocess.check_call(cmd)
                return {'device': device.get('id'), 'ok': True}
            except Exception as e:
                return {'device': device.get('id'), 'ok': False, 'error': str(e)}
        else:
            # API-based deploy not implemented here
            return {'device': device.get('id'), 'ok': False, 'error': 'unsupported_method'}

    def deploy_to_all_devices(self, package_path: str, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        results = []
        for d in self.edge_devices:
            res = self.deploy_to_edge(package_path, d)
            results.append(res)
            # record to registry if available and model_id provided
            if self.registry is not None and model_id is not None:
                try:
                    device_id = d.get('id') or d.get('host') or str(d)
                    status = 'ok' if res.get('ok') else 'failed'
                    self.registry.record_deployment(model_id, device_id, status=status)
                except Exception:
                    # do not fail deployment recording
                    pass
        return results
