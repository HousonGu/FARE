"""Fetch the PatchTST and SparseTSF source files listed in configs/upstream_sources.json."""
from pathlib import Path
import argparse
import json
import urllib.request

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sources = json.loads((HERE / 'configs/upstream_sources.json').read_text(encoding='utf-8'))
    for name, record in sources.items():
        for destination, item in record['files'].items():
            request = urllib.request.Request(item['url'], headers={'User-Agent': 'FARE-reproduction'})
            with urllib.request.urlopen(request, timeout=60) as stream:
                original = stream.read()
            content = original
            if destination.endswith('.py'):
                code = original.decode('utf-8-sig')
                if name == 'patchtst':
                    code = code.replace('from layers.', 'from vendor.patchtst.layers.')
                else:
                    code = code.replace('from layers.Embed import PositionalEmbedding', '')
                content = ('# Retail bundle adaptation: namespace imports only; forecasting computations unchanged.\n' + code).encode('utf-8')
            path = HERE / 'vendor' / name / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(name, record['repository'])
    for name in ['vendor', 'vendor/patchtst', 'vendor/patchtst/layers', 'vendor/sparsetsf']:
        (HERE / name / '__init__.py').write_text('', encoding='utf-8')
    (HERE / 'vendor/SOURCES.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
