"""Optional independent LP check: pip install scipy; python -m scripts.lp_oracle."""
import numpy as np
from scipy.optimize import linprog
from src.transport import certificate


def run():
    rng=np.random.default_rng(20261125);count=0;maxerr=0
    for _ in range(25):
        n=5; mu=rng.uniform(.1,.9,n); proxy=np.clip(mu+rng.uniform(-.12,.12,n),0,1);b=proxy-mu
        edges=[(i,j,abs(b[i]-b[j])+rng.uniform(0,.04)) for i in range(n) for j in range(i+1,n) if rng.random()<.5]
        anchors={0:(max(0,mu[0]-.03),min(1,mu[0]+.03))}
        D=certificate(proxy,proxy,edges,anchors)
        A=[];v=[]
        for i,j,r in edges:
            z=np.zeros(n);z[j]=1;z[i]=-1;A.extend([z,-z]);v.extend([proxy[j]-proxy[i]+r,proxy[i]-proxy[j]+r])
        bounds=[anchors.get(i,(0,1)) for i in range(n)]
        for i in range(n):
            for j in range(n):
                z=np.zeros(n);z[j]+=1;z[i]-=1
                lp=linprog(-z,A_ub=np.array(A) if A else None,b_ub=np.array(v) if v else None,bounds=bounds,method='highs')
                assert lp.success
                err=abs(-lp.fun-D[i,j]);maxerr=max(maxerr,err);assert err<1e-8;count+=1
    print('LP oracle comparisons:',count,'max absolute error:',maxerr)

if __name__=='__main__':run()
