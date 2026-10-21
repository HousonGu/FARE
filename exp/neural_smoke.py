"""Actual-data, actual-model smoke; development weeks only, never a formal result."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
from data_provider.panel import Panel
from backbones.neural import configure, run_backbone
from exp.neural_fusion import run_fusion
from exp.io import write_json, merge, sha, source_hashes


class SmokePanel:
    context = Panel.context
    keys = Panel.keys
    indices = Panel.indices
    truth = Panel.truth

    def __init__(self, original):
        self.original = original
        for name in ['folder', 'dataset', 'ids', 'id_rows', 'values', 'days', 'columns']:
            setattr(self, name, getattr(original, name))
        self._indices = {}
        fit = original.schedule['fit'][-6:]
        dev, test = original.schedule['dev'][:3], original.schedule['dev'][3:5]
        self.schedule = dict(train=fit[:-2], tune=fit[-2:], fit=fit, dev=dev, test=test)
        common = set(original.rows(fit[0]))
        for origin in fit+dev+test:
            common.intersection_update(original.rows(origin))
        self.selected = np.asarray(sorted(common)[:24])
        if len(self.selected) < 4:
            raise ValueError('Insufficient common real smoke series')
        self.expected = (len(test), len(self.selected)*7*len(test))

    def rows(self, origin):
        return self.selected

    def reference(self, origin):
        frame = self.original.reference(origin)
        frame = frame[frame.series_id.isin(self.ids[self.selected])].copy()
        return merge(self.keys(origin), frame)


def smoke(data_root, out, device, config, dataset='all'):
    config = copy.deepcopy(config)
    config.update(seeds=[config['seeds'][0]], batch_size=16, prediction_batch_size=24,
                  max_epochs=2, patience=2, cpu_threads=2)
    config['patchtst'].update(d_models=[32], patch_lengths=[7], learning_rates=[0.001])
    config['sparsetsf'].update(learning_rates=[0.001], weight_decays=[0.0])
    configure(config, device)
    stamp = dict(code=source_hashes(), config=config, device=device,
                 data={d: sha(Path(data_root)/d/'complete.json') for d in config['datasets']})
    manifest = Path(out)/'smoke/manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != stamp:
        raise ValueError('Smoke inputs changed; use a new smoke output directory')
    write_json(manifest, stamp)
    datasets = config['datasets'] if dataset == 'all' else [dataset]
    results = []
    for name in datasets:
        panel = SmokePanel(Panel(Path(data_root)/name))
        for kind in config['backbones']:
            directory = Path(out)/'smoke'/name
            run_backbone(panel, kind, directory/'models'/kind, device, config)
            seed = config['seeds'][0]
            result = run_fusion(panel, directory/'models'/kind/str(seed), directory/'fusion'/kind/str(seed), kind, seed, config)
            results.append(dict(dataset=name, backbone=kind, seed=seed, test_rows=panel.expected[1],
                source_development_origins_used_as_smoke_test=panel.schedule['test'],
                variants=list(result['overall']), metrics=result['overall']))
    report = dict(status='smoke_complete_not_formal_result', device=device, real_model_cases=len(results),
        note='Small subsets and original development weeks only. No formal-test results; do not put these metrics in manuscript.',
        config=config, code=stamp['code'], cases=results)
    write_json(Path(out)/'smoke/report.json', report)
    print('smoke complete:', len(results), 'real-data/model cases; not a formal result', flush=True)
    return report
