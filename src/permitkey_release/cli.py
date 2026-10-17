import argparse,json
from .common import HEADS,MASK,fresh,write_json
from . import generic_label_rule

def main():
    p=argparse.ArgumentParser(description='PermitKey frozen classification and aggregate release tools')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('summarize');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--all25',action='store_true')
    a=s.add_parser('figure');a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('aggregate');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--version',required=True);a.add_argument('--level',choices=['tract','jurisdiction'],default='tract')
    a=s.add_parser('classify');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--model',required=True);a.add_argument('--batch-size',type=int,default=2048)
    a=p.parse_args()
    if a.command=='summarize':
        from .validation import summarize
        write_json(a.output,summarize(a.input,a.all25))
    elif a.command=='figure':
        from .validation import figure
        figure(a.input,a.output)
    elif a.command=='aggregate':
        from .aggregate import aggregate
        print(json.dumps(aggregate(a.input,a.output,a.version,a.level),indent=2))
    else:
        from .frozen_model import load,predict
        if a.batch_size<1:raise ValueError('batch-size must be positive')
        output=fresh(a.output);bundle=load(a.model)
        with open(a.input,encoding='utf-8') as src,output.open('x',encoding='utf-8') as dst:
            batch=[]
            def flush():
                for rec,r in zip(batch,predict(bundle,batch)):
                    r['predictions']=generic_label_rule.apply(rec,r['predictions'])
                    r['label_bits']=sum(1<<HEADS.index(k) for k,v in r['predictions'].items() if v)
                    r['selected14_positive']=bool(r['label_bits']&MASK)
                    dst.write(json.dumps(r,allow_nan=False)+'\n')
                batch.clear()
            for line in src:
                if not line.strip():continue
                r=json.loads(line)
                if not isinstance(r.get('uid'),str) or not r['uid']:raise ValueError('Nonempty string uid required')
                for k in ['description','native_type','native_subtype']:
                    if r.get(k) is not None and not isinstance(r[k],str):raise ValueError(k+' must be text or null')
                batch.append(r)
                if len(batch)>=a.batch_size:flush()
            if batch:flush()

if __name__=='__main__':main()
