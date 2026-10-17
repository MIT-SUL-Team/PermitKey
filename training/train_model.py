"""Train the 25-head PermitKey checkpoint."""
import hashlib,json,time
from pathlib import Path
import duckdb,joblib,numpy as np,sklearn,pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
ROOT=Path('data')
OUT=ROOT/'model'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'candidate.joblib').exists():raise RuntimeError('Model exists; choose a new version rather than overwrite')
 cfgpath=Path(__file__).resolve().parents[1]/'config'/'training_concepts.json'
 cfg=json.loads(cfgpath.read_text());frozen=cfg['frozen_8'];added=cfg['new_17']
 train8=ROOT/'training8/train2.parquet'
 train25=ROOT/'natural25/train25_citysup_v2.parquet'
 vec=HashingVectorizer(n_features=2**18,ngram_range=(1,2),alternate_sign=False,norm='l2')
 con=duckdb.connect();con.execute("SET temp_directory='data/duckdb_temp'");con.execute("SET max_temp_directory_size='2GB'");con.execute("SET threads=1");con.execute("SET memory_limit='32GB'")
 queries={};models={};stats={};selected_frames={};start=time.time()
 for lane,training,concepts,feature in [('frozen8',train8,frozen,'feat'),('added17',train25,added,'descr')]:
  tech=[c['tech'] for c in concepts]
  terms=[f'label_{t}+flag_{t}' for t in tech] if lane=='frozen8' else [f'label_{t}' for t in tech]
  psum='('+'+'.join(terms)+')';path=str(training).replace('\\','/').replace("'","''")
  # Eight-head training query with a DuckDB reservoir sample of negatives.
  q=f"SELECT * FROM '{path}' WHERE {psum}>0 UNION ALL BY NAME SELECT * FROM '{path}' WHERE {psum}=0 USING SAMPLE 40000 ROWS (reservoir,7)"
  if lane=='added17':q=f"SELECT * FROM '{path}'"
  queries[lane]=q;df=con.execute(q).df();print(lane,'training_rows',len(df),flush=True)
  selected_frames[lane]=hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()).hexdigest()
  X=vec.transform(df[feature].fillna('').values)
  for c in concepts:
   t=c['tech'];y=df['label_'+t].values.astype(int);pos=int(y.sum())
   if pos<20 or pos==len(y):models[t]=None;stats[t]={'trained':False,'positives':pos};continue
   m=SGDClassifier(loss='log_loss',class_weight='balanced',alpha=1e-6,max_iter=15,tol=1e-3,random_state=0)
   m.fit(X,y);models[t]=m;stats[t]={'trained':True,'positives':pos,'training_rows':len(y),'feature':feature}
   print('trained',t,flush=True)
  del X,df
 bundle={'vectorizer':vec,'models':models,'concepts':frozen+added,'frozen_tech':[c['tech'] for c in frozen]}
 joblib.dump(bundle,OUT/'candidate.joblib',compress=3)
 manifest={'sklearn':sklearn.__version__,'duckdb':duckdb.__version__,'queries':queries,'heads':stats,'input_sha256':{p.name:sha(p) for p in [train8,train25,cfgpath]},'model_sha256':sha(OUT/'candidate.joblib'),'elapsed_seconds':time.time()-start,'score_semantics':'uncalibrated logistic model score; not ensemble probability'}
 manifest['selected_frame_ordered_hashes']=selected_frames
 manifest['query_threads']=1
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()

