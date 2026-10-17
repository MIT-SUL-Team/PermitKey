"""Check that the checkpoint reproduces the saved scores for the reviewed permits."""
import json,sys,platform,importlib.metadata
from pathlib import Path
from permitkey_release.frozen_model import load,predict
root=Path(__file__).resolve().parents[1];rows=json.loads((root/'validation/reviewed_inputs.json').read_text());tasks=json.loads((root/'validation/human_completed_tasks.json').read_text());model=load(sys.argv[1]);out={r['uid']:r for r in predict(model,rows)}
error=0.;n=0
for t in tasks:
 for k,v in t['scores'].items():
  error=max(error,abs(v-out[t['uid']]['probabilities'][k]));n+=1
report={'reviewed_records':len(rows),'task_score_comparisons':n,'max_probability_error':error,'python':platform.python_version(),'versions':{k:importlib.metadata.version(k) for k in ['duckdb','numpy','scikit-learn','joblib']},'checkpoint_training_sklearn':'1.9.1'}
print(json.dumps(report,indent=2));assert error<1e-10
