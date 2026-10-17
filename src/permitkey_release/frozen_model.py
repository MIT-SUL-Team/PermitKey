"""Inference with the released PermitKey checkpoint (25 heads; 14 released categories)."""
import hashlib
from pathlib import Path
import numpy as np
import joblib

MODEL_SHA='d053e0d36b10ec302855d9392c59150dba5fccce0bdf8ac529bc8ca5b83b6eff'
def load(path):
 path=Path(path)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Checkpoint does not match the released model hash')
 b=joblib.load(path)
 if len(b['concepts'])!=25:raise ValueError('Expected all 25 heads')
 return b
def predict(bundle,records):
 """Predict all 25 heads for a batch of records."""
 desc=[r.get('description') or '' for r in records]
 typ=[r.get('native_type') or '' for r in records]
 sub=[r.get('native_subtype') or '' for r in records]
 x8=bundle['vectorizer'].transform([(d+' | '+t+' | '+s).lower() for d,t,s in zip(desc,typ,sub)])
 x17=bundle['vectorizer'].transform(desc)
 result=[]
 valid8=np.array([bool((d+t+s).strip()) for d,t,s in zip(desc,typ,sub)])
 valid17=np.array([bool(d.strip()) for d in desc])
 columns={}
 for concept in bundle['concepts']:
  name,tech=concept['natural25'],concept['tech'];m=bundle['models'][tech]
  if m is None:raise ValueError('Missing fitted head: '+name)
  original=tech in bundle['frozen_tech']
  scores=m.predict_proba(x8 if original else x17)[:,list(m.classes_).index(1)]
  columns[name]=(scores,(scores>.5)&(valid8 if original else valid17),original)
 for i,r in enumerate(records):
  result.append({'uid':r['uid'],'probabilities':{k:float(v[0][i]) for k,v in columns.items()},
   'predictions':{k:int(v[1][i]) for k,v in columns.items()},
   'production_available':{k:bool(valid8[i] if v[2] else valid17[i]) for k,v in columns.items()},
   'evaluation_eligible':{k:bool(valid17[i]) and not r.get('source_conflict',False) and (not v[2] or r.get('source_subtype_verified',False)) for k,v in columns.items()},
   'model_id':'permitkey-model-v1'})
 return result
