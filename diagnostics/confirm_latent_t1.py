"""Research-only parameter-matched T=1 latent/register confirmation."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

W=4

def digits(v): return [int(c) for c in f"{v:0{W}d}"][::-1]
def moduli(seed=0):
    primes=(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47)
    vals=sorted({p*q for i,p in enumerate(primes) for q in primes[i+1:] if 21<=p*q<=99})
    return random.Random(seed).sample(vals,26)
def split_rows(ms, split_seed=45):
    train=[]; held=[]
    for n in ms:
        xs=list(range(n)); random.Random(split_seed+n).shuffle(xs); cut=int(.8*len(xs))
        for x in xs[:cut]:
            y=(x*x)%n; train.append((n,x,digits(y)))
        for x in xs[cut:]:
            y=(x*x)%n; held.append((n,x,digits(y)))
    return train,held
def rows(ms):
    return [(n,x,digits((x*x)%n)) for n in ms for x in range(n)]

class MatchedBase(nn.Module):
    def __init__(self,d=128):
        super().__init__(); self.d=d
        self.digit=nn.Embedding(10,32); self.place=nn.Embedding(W,32)
        self.enc=nn.Sequential(nn.Linear(2*W*32,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
        self.nenc=nn.Sequential(nn.Linear(W*32,d),nn.LayerNorm(d),nn.GELU())
        self.step=nn.GRUCell(2*d,d)
        self.dec=nn.Sequential(nn.Linear(d+32,d),nn.GELU(),nn.Linear(d,10))
    def encode_inputs(self,n,x):
        pos=torch.arange(W,device=n.device); pe=self.place(pos)
        ne=(self.digit(n)+pe).flatten(1); xe=(self.digit(x)+pe).flatten(1)
        return self.enc(torch.cat((ne,xe),-1)),self.nenc(ne),pe
    def decode(self,h,pe):
        return self.dec(torch.cat((h,pe[None].expand(h.shape[0],-1,-1)),-1))

class GlobalLatent(MatchedBase):
    def forward(self,n,x):
        h,ns,pe=self.encode_inputs(n,x); h=self.step(torch.cat((h,ns),-1),h)
        return self.decode(h[:,None].expand(-1,W,-1),pe),h

class RegisterControl(MatchedBase):
    def forward(self,n,x):
        h,ns,pe=self.encode_inputs(n,x); h=h[:,None].expand(-1,W,-1)
        inp=torch.cat((h,ns[:,None].expand(-1,W,-1)),-1)
        h=self.step(inp.reshape(-1,2*self.d),h.reshape(-1,self.d)).reshape_as(h)
        return self.decode(h,pe),h

def batch(data,size,step,device):
    z=[data[(step*size+i)%len(data)] for i in range(size)]
    return (torch.tensor([digits(r[0]) for r in z],device=device),torch.tensor([digits(r[1]) for r in z],device=device),torch.tensor([r[2] for r in z],device=device))
@torch.no_grad()
def evaluate(model,data,device):
    model.eval(); good=[]
    for i in range(0,len(data),1024):
        z=data[i:i+1024]; n=torch.tensor([digits(r[0]) for r in z],device=device); x=torch.tensor([digits(r[1]) for r in z],device=device); y=torch.tensor([r[2] for r in z],device=device)
        good.extend(((model(n,x)[0].argmax(-1)==y).all(-1)).tolist())
    return sum(good)/len(good)
def train(model,data,held,unseen,seconds,seed,device):
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,betas=(.9,.95),weight_decay=.05)
    start=time.monotonic(); step=0; curve=[]
    while time.monotonic()-start<seconds:
        n,x,y=batch(data,512,step,device); model.train(); logits,_=model(n,x)
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step+=1
        if step==1 or step%100==0: curve.append({'step':step,'seconds':round(time.monotonic()-start,2),'loss':float(loss.detach()),'train_exact':float((logits.argmax(-1)==y).all(-1).float().mean())})
    return {'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'steps':step,'seconds':time.monotonic()-start,'curve':curve,'train_exact':evaluate(model,data,device),'held_out_x_exact':evaluate(model,held,device),'unseen_N_exact':evaluate(model,unseen,device)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seconds',type=float,default=120); ap.add_argument('--device',default='cuda'); args=ap.parse_args(); device=torch.device(args.device)
    ms=moduli(0); train_ms,test_ms=ms[:18],ms[18:]; train_rows,held=split_rows(train_ms); unseen=rows(test_ms); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    all_reports={}
    for seed in (0,1):
        torch.manual_seed(seed); random.seed(seed)
        for name,cls in [('global_latent',GlobalLatent),('register_control',RegisterControl)]:
            torch.manual_seed(seed); model=cls().to(device)
            rep=train(model,train_rows,held,unseen,args.seconds,seed,device); all_reports[f'{name}_seed{seed}']=rep
            if name=='global_latent': torch.save({'state_dict':model.state_dict(),'seed':seed,'parameters':rep['parameters']},out/f'latent_seed{seed}.pt')
            print(name,seed,json.dumps(rep),flush=True)
    report={'classification':'RESEARCH ONLY — final-label T=1; exact parameter-matched state-interface comparison','train_moduli':train_ms,'unseen_moduli':test_ms,'train_rows':len(train),'held_out_x_rows':len(held),'unseen_rows':len(unseen),'reports':all_reports}
    (out/'eval_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
