import os, sys
sys.path.insert(0, '<analysis>/pipeline/e2e_joint')
import thermal_bank_pim as B
import thermal_pim_sim as TP
G = os.path.abspath('geom_s070')
m = B.load(G); cols = TP.columns(G)
cases = {'uni': dict(maps=[(200.0, lambda b: True)]), 'array': dict(maps=[(200.0, lambda b: b['kind'] == 'ar')]),
         'gpu': dict(gpu=1000.0)}
for nm, kw in cases.items():
    r = B.row(cols, m, **kw)
    os.makedirs(os.path.join('runs', nm), exist_ok=True)
    with open(os.path.join('runs', nm, nm + '.ptrace'), 'w') as f:
        f.write('\t'.join(cols) + '\n' + '\t'.join(f'{x:.6f}' for x in r) + '\n')
    print(nm, round(sum(r), 3))
