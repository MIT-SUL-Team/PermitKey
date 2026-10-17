import json
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name('categories.json').read_text())
HEADS = CONFIG['heads']
SELECTED = CONFIG['selected14']
MASK = sum(1 << HEADS.index(k) for k in SELECTED)

def fresh(path):
    p = Path(path)
    if p.exists():
        raise FileExistsError(f'Refusing to overwrite {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def literal(value):
    return "'" + str(value).replace("'", "''") + "'"

def write_json(path, obj):
    with fresh(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
