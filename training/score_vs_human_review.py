#!/usr/bin/env python3
"""Compare the linear model with a fine-tuned sentence encoder on the human reference set."""
import argparse, json, numpy as np, duckdb, csv, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

TECH=["solar","ev_charger","battery","heat_pump","panel","reroof","pool_spa","demolition"]
FT_WINS={"ev_charger","heat_pump","demolition"}

def prf(pred,y):
    tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
    P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;F=2*P*R/(P+R) if P+R else 0
    return P,R,F,tp,fp,fn

def boot_f1(pred,y,n=5000,seed=20260903):
    rng=np.random.RandomState(seed); m=len(y); fs=[]
    for _ in range(n):
        idx=rng.randint(0,m,m); p,yy=pred[idx],y[idx]
        tp=((p==1)&(yy==1)).sum();fp=((p==1)&(yy==0)).sum();fn=((p==0)&(yy==1)).sum()
        P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;fs.append(2*P*R/(P+R) if P+R else 0)
    return float(np.percentile(fs,2.5)),float(np.percentile(fs,97.5))

class Net(nn.Module):
    def __init__(s,name,n):super().__init__();s.enc=AutoModel.from_pretrained(name);s.head=nn.Linear(s.enc.config.hidden_size,n)
    def forward(s,ids,mask):
        h=s.enc(input_ids=ids,attention_mask=mask).last_hidden_state
        return s.head((h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp(min=1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True);ap.add_argument("--gold",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs",type=int,default=3);ap.add_argument("--bs",type=int,default=256);ap.add_argument("--maxlen",type=int,default=64)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"

    con=duckdb.connect();con.execute("PRAGMA threads=8")
    psum="("+"+".join([f"label_{t}+flag_{t}" for t in TECH])+")"
    df=con.execute(f"""SELECT * FROM '{a.train}' WHERE {psum}>0 UNION ALL BY NAME
        SELECT * FROM '{a.train}' WHERE {psum}=0 USING SAMPLE 25000 ROWS (reservoir,42)""").df()
    # Reference set
    gtxt=[];G={t:[] for t in TECH}
    with open(a.gold,encoding="utf-8-sig",newline="") as h:
        for r in csv.DictReader(h):
            gtxt.append(r["description"])
            for t in TECH: G[t].append(int(r[f"gold_{t}"]))
    G={t:np.array(v) for t,v in G.items()}; print(f"train={len(df)} gold={len(gtxt)} dev={dev}",flush=True)

    # Linear model
    vec=HashingVectorizer(n_features=2**18,ngram_range=(1,2),alternate_sign=False,norm='l2')
    Xtr=vec.transform(df["descr"].fillna("").values); Xg=vec.transform(gtxt)
    linpred={}
    for t in TECH:
        y=df[f"label_{t}"].values.astype(int)
        m=SGDClassifier(loss='log_loss',class_weight='balanced',alpha=1e-6,max_iter=15,tol=1e-3,random_state=0);m.fit(Xtr,y)
        linpred[t]=m.predict(Xg)

    # Fine-tuned model with per-category thresholds
    txt=df["descr"].fillna("").tolist(); Y=np.stack([df[f"label_{t}"].values.astype(np.float32) for t in TECH],1)
    rng=np.random.RandomState(1234); idx=np.arange(len(df)); rng.shuffle(idx)
    nval=int(0.1*len(idx)); vi=idx[:nval]; ti=idx[nval:]
    tok=AutoTokenizer.from_pretrained(a.model)
    def tokz(lst): e=tok(lst,truncation=True,max_length=a.maxlen,padding="max_length",return_tensors="pt");return e["input_ids"],e["attention_mask"]
    ids,mask=tokz(txt); gids,gmask=tokz(gtxt); ytt=torch.tensor(Y)
    dl=DataLoader(TensorDataset(ids[ti],mask[ti],ytt[ti]),batch_size=a.bs,shuffle=True)
    net=Net(a.model,len(TECH)).to(dev)
    pos=ytt[ti].sum(0);pw=((len(ti)-pos)/pos.clamp(min=1)).clamp(max=50).to(dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw);opt=torch.optim.AdamW(net.parameters(),lr=2e-5);net.train()
    for ep in range(a.epochs):
        tl=0
        for bi,bm,by in dl:
            bi,bm,by=bi.to(dev),bm.to(dev),by.to(dev);opt.zero_grad();l=lossf(net(bi,bm),by);l.backward();opt.step();tl+=l.item()
        print(f"ep{ep+1} loss {tl/len(dl):.4f}",flush=True)
    def sc(ii,mm):
        net.eval();o=[]
        with torch.no_grad():
            for s in range(0,len(ii),512):o.append(torch.sigmoid(net(ii[s:s+512].to(dev),mm[s:s+512].to(dev))).cpu().numpy())
        return np.concatenate(o)
    sv=sc(ids[vi],mask[vi]);sg=sc(gids,gmask);yv=Y[vi].astype(int)
    grid=np.arange(0.05,0.96,0.05); ftpred={}
    for j,t in enumerate(TECH):
        best_t,best=0.5,-1
        for th in grid:
            _,_,F,_,_,_=prf((sv[:,j]>th).astype(int),yv[:,j])
            if F>best:best,best_t=F,th
        ftpred[t]=(sg[:,j]>best_t).astype(int)

    rows={}; models={"linear":linpred,"finetuned_cal":ftpred}
    for name,pred in models.items():
        rows[name]={}
        for t in TECH:
            P,R,F,tp,fp,fn=prf(pred[t],G[t]); lo,hi=boot_f1(pred[t],G[t])
            rows[name][t]=dict(P=round(P,3),R=round(R,3),F1=round(F,3),ci=[round(lo,3),round(hi,3)],n_pos=int(G[t].sum()))
    # Better of the two per category
    rows["best_of"]={}
    for t in TECH:
        pred=ftpred[t] if t in FT_WINS else linpred[t]
        P,R,F,tp,fp,fn=prf(pred,G[t]); lo,hi=boot_f1(pred,G[t])
        rows["best_of"][t]=dict(P=round(P,3),R=round(R,3),F1=round(F,3),ci=[round(lo,3),round(hi,3)],src=("ft" if t in FT_WINS else "linear"))
    # Micro-averaged
    def micro(pred_of):
        tp=fp=fn=0
        for t in TECH:
            p,y=pred_of[t],G[t]; tp+=((p==1)&(y==1)).sum();fp+=((p==1)&(y==0)).sum();fn+=((p==0)&(y==1)).sum()
        P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;return round(2*P*R/(P+R) if P+R else 0,3)
    bestpred={t:(ftpred[t] if t in FT_WINS else linpred[t]) for t in TECH}
    mic={"linear":micro(linpred),"finetuned_cal":micro(ftpred),"best_of":micro(bestpred)}
    out={"rows":rows,"micro":mic}
    Path(a.out).write_text(json.dumps(out,indent=1))
    print("\n=== vs HUMAN GOLD (F1 [ci]) ===")
    print(f"{'concept':<12}{'n+':>4}{'linear':>16}{'finetuned':>16}{'best_of':>16}")
    for t in TECH:
        l=rows['linear'][t];f=rows['finetuned_cal'][t];b=rows['best_of'][t]
        print(f"{t:<12}{l['n_pos']:>4}  {l['F1']:.2f}[{l['ci'][0]:.2f}-{l['ci'][1]:.2f}]  {f['F1']:.2f}[{f['ci'][0]:.2f}-{f['ci'][1]:.2f}]  {b['F1']:.2f}[{b['ci'][0]:.2f}-{b['ci'][1]:.2f}]({b['src']})")
    print(f"\nMICRO  linear={mic['linear']}  finetuned={mic['finetuned_cal']}  best_of={mic['best_of']}")
    print("RB");print(json.dumps(out));print("RE")

if __name__=="__main__": main()
