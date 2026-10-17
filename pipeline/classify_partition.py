"""Classify a range of source contributions with the released checkpoint (for parallel runs)."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import joblib,numpy as np,pyarrow as pa,pyarrow.parquet as pq
import master_normalization_copy as norm
ROOT=Path('data')

def extracted_field(raw,tab,names):
 value=norm.select_column_array(raw,names)
 return norm.fill_from_raw(value,tab['raw_json'],names,tab['raw_json'].to_pylist())

def strict_issue_datetime(raw,tab):
 return norm.normalize_datetime(extracted_field(raw,tab,['issue_date','issued_date','date_issued','issuedate','issued','issuance_date','permit_issued_date','issue_dt','issued_datetime','issued_on']))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rows',type=int,default=1000);ap.add_argument('--full',action='store_true');ap.add_argument('--all-contributions',action='store_true');ap.add_argument('--contribution',type=int,default=0);ap.add_argument('--start-contribution',type=int,required=True);ap.add_argument('--stop-contribution',type=int,required=True);a=ap.parse_args()
 if not a.full or not 0<=a.start_contribution<a.stop_contribution<=85:raise ValueError('Valid full-run partition required')
 verify=json.loads((ROOT/'00_admin/LOCAL_SOURCE_VERIFICATION.json').read_text())
 if not verify['ok']:raise RuntimeError('Source verification failed')
 ledger=json.loads((ROOT/'00_admin/LOCAL_CONTRIBUTION_LEDGER.json').read_text())
 modelroot=ROOT/'model';manifest=json.loads((modelroot/'manifest.json').read_text())
 adapter_hash=hashlib.sha256(Path(__file__).with_name('classify_copies.py').read_bytes()+Path(norm.__file__).read_bytes()).hexdigest()
 bundle=joblib.load(modelroot/'candidate.joblib');vec=bundle['vectorizer'];models=bundle['models'];concepts=bundle['concepts']
 outroot=ROOT/'04_classifications'/('text_candidate_v1' if a.full else ('smoke_all_text_candidate_v2' if a.all_contributions else 'smoke_text_candidate_v2'))
 outroot.mkdir(exist_ok=True);summary=[];start=time.time()
 for ci,c in enumerate(ledger['contributions']):
  if not a.start_contribution<=ci<a.stop_contribution:continue
  if not a.full and not a.all_contributions and ci!=a.contribution:continue
  contribution_start=time.time()
  src=Path(c['path']).resolve()
  if not src.is_relative_to((ROOT/'01_sources').resolve()):raise RuntimeError('Source outside immutable copied tree')
  files=sorted(src.rglob('*.parquet')) if src.is_dir() else [src]
  dest=outroot/norm.slug(c['materialization']);dest.mkdir(exist_ok=True);ordinal=0;batchid=0;processed=0
  for file in files:
   for batch in pq.ParquetFile(file).iter_batches(batch_size=32768):
    if not a.full and processed>=a.rows:break
    if not a.full:batch=batch.slice(0,min(batch.num_rows,a.rows-processed))
    target=dest/f'part-{batchid:06d}.parquet';marker=target.with_suffix('.json')
    if marker.exists():
     old=json.loads(marker.read_text())
     if old['model_sha256']!=manifest['model_sha256'] or old.get('adapter_sha256')!=adapter_hash or old['rows']!=batch.num_rows or old['row_start']!=ordinal or not target.exists():raise RuntimeError('Resume mismatch')
     ordinal+=batch.num_rows;processed+=batch.num_rows;batchid+=1;continue
    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Stopped before consuming 3 GiB disk reserve')
    tab=norm.build_batch(batch,c,file,[ROOT/'01_sources/orcd_pool',ROOT/'01_sources/orcd_home_mirror'],ordinal)
    desc=[v or '' for v in tab['description'].to_pylist()]
    raw=pa.Table.from_batches([batch])
    typ=extracted_field(raw,tab,['record_type_original','record_type','permit_type','work_type','application_type','type','original_native_label'])
    types=[v or '' for v in typ.to_pylist()]
    sub=extracted_field(raw,tab,['record_subtype_original','record_subtype','permit_subtype','subtype','sub_type'])
    subtype=[v or '' for v in sub.to_pylist()]
    features=[(d+' | '+t+' | '+s).lower() for d,t,s in zip(desc,types,subtype)]
    x8=vec.transform(features);x17=vec.transform(desc);n=len(desc)
    valid8=np.array([bool((d+t+s).strip()) for d,t,s in zip(desc,types,subtype)])
    valid17=np.array([bool(d.strip()) for d in desc]);bits=np.zeros(n,dtype=np.uint32);avail=np.zeros(n,dtype=np.uint32)
    positive_scores=[[] for _ in range(n)]
    for k,cpt in enumerate(concepts):
     tech=cpt['tech'];m=models[tech]
     if m is None:continue
     is8=tech in bundle['frozen_tech'];valid=valid8 if is8 else valid17
     scores=m.predict_proba(x8 if is8 else x17)[:,list(m.classes_).index(1)]
     pred=(scores>0.5)&valid;bits[pred]|=np.uint32(1<<k);avail[valid]|=np.uint32(1<<k)
     for row in np.flatnonzero(pred):positive_scores[row].append(float(scores[row]))
    # Keep issue dates separate from approval dates.
    out=tab.select(['record_id','source_contribution_id','source_file','source_row_ordinal','native_source_key','application_datetime_utc','latitude','longitude','coordinate_provenance'])
    out=out.append_column('issue_datetime_utc',strict_issue_datetime(raw,tab))
    out=out.append_column('issue_or_approval_datetime_utc',tab['issue_datetime_utc'])
    out=out.append_column('native_jurisdiction',extracted_field(raw,tab,['jurisdiction','permitting_jurisdiction','permitting_authority']))
    out=out.append_column('native_source_url',extracted_field(raw,tab,['source_url']))
    out=out.append_column('model_positive_bits',pa.array(bits)).append_column('model_available_bits',pa.array(avail))
    out=out.append_column('positive_model_scores',pa.array(positive_scores,type=pa.list_(pa.float32())))
    out=out.append_column('ensemble_available',pa.array([False]*n)).append_column('classification_lane',pa.array(['learned_text_candidate']*n))
    part=target.with_suffix('.parquet.partial');pq.write_table(out,part,compression='zstd');part.replace(target)
    info={'rows':n,'row_start':ordinal,'source_file':str(file),'model_sha256':manifest['model_sha256'],'bytes':target.stat().st_size,'lane':'text_only_no_compatible_flags','feature_adapter':'v1_explicit_type_and_subtype_fields_needs_source_parity_review'}
    info['adapter_sha256']=adapter_hash
    marker.write_text(json.dumps(info));ordinal+=n;processed+=n;batchid+=1
    if batchid%10==0:
     progress={'active_contribution':ci,'contributions_completed':len(summary),'rows':sum(x['rows'] for x in summary)+processed,'active_contribution_rows':processed,'elapsed_seconds':time.time()-start}
     print(json.dumps(progress),flush=True)
     (outroot/f'progress_{a.start_contribution}_{a.stop_contribution}.json').write_text(json.dumps(progress))
   if not a.full and processed>=a.rows:break
  summary.append({'contribution':ci,'rows':processed,'source_records':c['records'],'expected':c['records'] if a.full else min(a.rows,c['records']),'output_bytes':sum(p.stat().st_size for p in dest.glob('part-*.parquet')),'elapsed_seconds':time.time()-contribution_start})
  print(json.dumps(summary[-1]),flush=True)
  (outroot/f'progress_{a.start_contribution}_{a.stop_contribution}.json').write_text(json.dumps({'contributions_completed':len(summary),'rows':sum(x['rows'] for x in summary),'elapsed_seconds':time.time()-start}))
  if a.full and processed!=c['records']:raise RuntimeError('Contribution row mismatch')
 result={'status':'candidate_only','entries':summary,'rows':sum(x['rows'] for x in summary),'elapsed_seconds':time.time()-start,'head_order':[c['natural25'] for c in concepts],'model_sha256':manifest['model_sha256'],'ensemble_available':False}
 result['adapter_sha256']=adapter_hash
 (outroot/f'run_summary_{a.start_contribution}_{a.stop_contribution}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
