"""City-held-out evaluation of the original eight heads."""
import duckdb, json, os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
SC="data/training8"
con=duckdb.connect(); con.execute("PRAGMA threads=8")
TECH=["solar","ev_charger","battery","heat_pump","panel","reroof","pool_spa","demolition"]
psum="("+"+".join([f"label_{t}+flag_{t}" for t in TECH])+")"
# All positives plus a capped sample of negatives.
df=con.execute(f"""
 SELECT * FROM '{SC}/train.parquet' WHERE {psum}>0
 UNION ALL BY NAME
 SELECT * FROM '{SC}/train.parquet' WHERE {psum}=0 USING SAMPLE 25000 ROWS (reservoir, 42)
""").df()
print("subsampled rows",len(df),flush=True)
cities=sorted(df['city'].unique().tolist())
vec=HashingVectorizer(n_features=2**18, ngram_range=(1,2), alternate_sign=False, norm='l2')
X=vec.transform(df['descr'].fillna('').values)
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    return dict(precision=round(P,3),recall=round(R,3),f1=round(2*P*R/(P+R),3) if P+R else 0.0,support=tp+fn)
out={}; raw={k:[0,0,0] for k in['flag','model','ensemble']}
for t in TECH:
    y=df[f'label_{t}'].values.astype(int); flag=df[f'flag_{t}'].values.astype(int)
    acc={k:[0,0,0] for k in['flag','model','ensemble']}
    for c in cities:
        te=(df['city']==c).values; tr=~te
        if int(y[tr].sum())<20 or int(y[te].sum())<5: continue
        clf=SGDClassifier(loss='log_loss',class_weight='balanced',alpha=1e-6,max_iter=12,tol=1e-3,random_state=0)
        clf.fit(X[tr],y[tr]); pm=clf.predict(X[te]); yf=flag[te]; yt=y[te]; pe=((pm==1)|(yf==1)).astype(int)
        for name,pred in[('flag',yf),('model',pm),('ensemble',pe)]:
            tp=int(((pred==1)&(yt==1)).sum()); fp=int(((pred==1)&(yt==0)).sum()); fn=int(((pred==0)&(yt==1)).sum())
            acc[name][0]+=tp;acc[name][1]+=fp;acc[name][2]+=fn; raw[name][0]+=tp;raw[name][1]+=fp;raw[name][2]+=fn
    out[t]={k:prf(*v) for k,v in acc.items()}
    print(t,"flag_f1",out[t]['flag']['f1'],"-> ens_f1",out[t]['ensemble']['f1'],flush=True)
out["MICRO_ALL"]={k:prf(*v) for k,v in raw.items()}
open(f"{SC}/ml_recall.json","w").write(json.dumps(out,indent=1))
print("RB"); print(json.dumps(out,indent=1)); print("RE")
