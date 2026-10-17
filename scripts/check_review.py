"""Reproduce the 14-category human-review micro/macro F1 from task-level labels and predictions."""
import json,sys
from pathlib import Path
cats=json.loads((Path(__file__).resolve().parents[1]/'config/categories.json').read_text())['selected14']
rows=json.loads(Path(sys.argv[1]).read_text());t={k:[0,0,0] for k in cats}
for r in rows:
 for k,v in r['truth'].items():
  if k not in t:continue
  p=r['prediction'][k];t[k][0]+=int(bool(p) and bool(v));t[k][1]+=int(bool(p) and not v);t[k][2]+=int(not p and bool(v))
tp,fp,fn=[sum(x[i] for x in t.values()) for i in range(3)]
f=lambda a,b,c:2*a/(2*a+b+c) if 2*a+b+c else 0
result={'tasks':len(rows),'groups':len({r['group_id'] for r in rows}),'micro_f1':f(tp,fp,fn),'macro_f1':sum(f(*x) for x in t.values())/len(cats),'confusion':t}
assert abs(result['micro_f1']-0.7991746905089409)<0.00001
print(json.dumps(result,indent=2))
