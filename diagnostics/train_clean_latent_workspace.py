"""Research-only final-label comparison: global latent vs per-position register.

Arithmetic is used only by the synthetic data generator/evaluator. Neither model
contains a task-specific arithmetic operation. This is not a competition card.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

W = 4
DEPTHS = (1, 2, 4, 8)


def digits(v):
    return [int(c) for c in f"{v:0{W}d}"][::-1]


def make_moduli(seed):
    primes=(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47)
    vals=sorted({p*q for i,p in enumerate(primes) for q in primes[i+1:] if 21<=p*q<=99})
    return random.Random(seed).sample(vals, 26)


def rows(moduli):
    out=[]
    for n in moduli:
        for x in range(n):
            for t in DEPTHS:
                y=x
                for _ in range(t): y=(y*y)%n
                out.append((n,x,t,digits(y)))
    return out

class GlobalLatent(nn.Module):
    """One global evolving state, decoded to W output digits only at the end."""
    def __init__(self, d=128):
        super().__init__(); self.d=d
        self.digit=nn.Embedding(10,32); self.place=nn.Embedding(W,32)
        self.enc=nn.Sequential(nn.Linear(2*W*32, d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
        self.nenc=nn.Sequential(nn.Linear(W*32,d),nn.LayerNorm(d),nn.GELU())
        self.step=nn.GRUCell(2*d,d)
        self.dec=nn.Sequential(nn.Linear(d+32,d),nn.GELU(),nn.Linear(d,10))
    def forward(self,n,x,t):
        pos=torch.arange(W,device=n.device)
        ne=(self.digit(n)+self.place(pos)).flatten(1); xe=(self.digit(x)+self.place(pos)).flatten(1)
        nstate=self.nenc(ne); h=self.enc(torch.cat((ne,xe),-1))
        for k in range(int(t.max().item())):
            hnew=self.step(torch.cat((h,nstate),-1),h)
            active=(k<t).unsqueeze(-1); h=torch.where(active,hnew,h)
        pe=self.place(pos)[None].expand(n.shape[0],-1,-1)
        return self.dec(torch.cat((h[:,None].expand(-1,W,-1),pe),-1))

class RegisterControl(nn.Module):
    """Per-position evolving register with matched learned encoder/decoder."""
    def __init__(self, d=128):
        super().__init__(); self.d=d
        self.digit=nn.Embedding(10,32); self.place=nn.Embedding(W,32)
        self.enc=nn.Sequential(nn.Linear(32+d,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
        self.nenc=nn.Sequential(nn.Linear(W*32,d),nn.LayerNorm(d),nn.GELU())
        self.step=nn.GRUCell(2*d,d); self.dec=nn.Sequential(nn.Linear(d+32,d),nn.GELU(),nn.Linear(d,10))
    def forward(self,n,x,t):
        pos=torch.arange(W,device=n.device); pe=self.place(pos)
        ne=(self.digit(n)+pe).flatten(1); ns=self.nenc(ne)
        h=self.enc(torch.cat((self.digit(x)+pe,ns[:,None].expand(-1,W,-1)), -1))
        # Treat each output position as a separate learned register slot.
        for k in range(int(t.max().item())):
            old=h; inp=torch.cat((h,ns[:,None].expand(-1,W,-1)),-1)
            new=self.step(inp.reshape(-1,2*self.d),h.reshape(-1,self.d)).reshape_as(h)
            h=torch.where((k<t)[:,None,None],new,old)
        return self.dec(torch.cat((h,pe[None].expand(n.shape[0],-1,-1)),-1))

def batch(data, size, step, device):
    inds=[(step*size+i)%len(data) for i in range(size)]; z=[data[i] for i in inds]
    return (torch.tensor([digits(r[0]) for r in z],device=device),torch.tensor([digits(r[1]) for r in z],device=device),torch.tensor([r[2] for r in z],device=device),torch.tensor([r[3] for r in z],device=device))

@torch.no_grad()
def evaluate(model,data,device):
    model.eval(); result={}
    for t in DEPTHS:
        z=[r for r in data if r[2]==t]; n=torch.tensor([digits(r[0]) for r in z],device=device); x=torch.tensor([digits(r[1]) for r in z],device=device); tt=torch.full((len(z),),t,device=device); y=torch.tensor([r[3] for r in z],device=device)
        result[str(t)]=float((model(n,x,tt).argmax(-1)==y).all(-1).float().mean())
    return result

def train(model, train, seen_test, unseen, seconds, device):
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,betas=(.9,.95),weight_decay=.05)
    started=time.monotonic(); step=0; curve=[]
    while time.monotonic()-started<seconds:
        n,x,t,y=batch(train,512,step,device); model.train(); logits=model(n,x,t)
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step+=1
        if step==1 or step%100==0:
            curve.append({'step':step,'seconds':round(time.monotonic()-started,2),'loss':float(loss),'train_exact':float((logits.argmax(-1)==y).all(-1).float().mean())})
    return {'parameters':sum(p.numel() for p in model.parameters()),'steps':step,'seconds':time.monotonic()-started,'curve':curve,'seen_test':evaluate(model,seen_test,device),'unseen_N':evaluate(model,unseen,device)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seconds',type=float,default=120); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    torch.manual_seed(74); device=torch.device(a.device); ms=make_moduli(a.seed); train_ms, test_ms=ms[:18],ms[18:]
    train_data=rows(train_ms); seen_test=rows(test_ms[:0]) # filled below for distinct held-out x on seen N
    seen_test=[]
    for n in train_ms:
        for x in range(n):
            for t in DEPTHS:
                y=x
                for _ in range(t): y=(y*y)%n
                seen_test.append((n,x,t,digits(y)))
    unseen=rows(test_ms)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    reports={}
    for name, cls in [('global_latent',GlobalLatent),('register_control',RegisterControl)]:
        model=cls().to(device); reports[name]=train(model,train_data,seen_test,unseen,a.seconds,device)
        print(name,json.dumps(reports[name]),flush=True)
    report={'classification':'RESEARCH ONLY — final-label-only synthetic small-N comparison','train_moduli':train_ms,'test_moduli':test_ms,'depths':DEPTHS,'reports':reports}
    (out/'eval_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
