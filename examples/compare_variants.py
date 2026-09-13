"""Run the four geometries with matched illustrative evidence parameters."""
import numpy as np
from ta import TA


def main():
    rng=np.random.RandomState(2026)
    members=rng.randint(0,4,(40,10))
    for name,delta,eta in [('Quad-SB',0,1),('Sym-TA',.25,1),
                           ('CapQ',.25,0),('TA',.25,.5)]:
        model=TA(3,lambda_=1e-3,gamma=2,delta=delta,eta=eta)
        model.fit_predict(members)
        print(name, 'delta=',delta,'eta=',eta,'sizes=',model.cluster_sizes_.tolist())


if __name__=='__main__':
    main()
