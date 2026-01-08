import meditation_config as mc
from collections import defaultdict
import numpy as np

def main():
    vals = defaultdict(list)
    for k, v in mc.DEFAULTS.items():
        vals[(str(type(v)), repr(v))].append(k)
    dups = {k: ks for k, ks in vals.items() if len(ks) > 1}
    print('DEFAULTS exact-duplicate groups count:', len(dups))
    for (t, val), ks in dups.items():
        print(ks, '=>', val)

    aip_fields = set(mc.ActInfParams.__annotations__.keys())
    def_keys = set(mc.DEFAULTS.keys())
    print('\nActInfParams ∩ DEFAULTS:', aip_fields & def_keys)

    states = mc.NETWORK_PROFILES['state_expected_profiles']
    print('\nState profile linear fits (a,b,resid):')
    for state, vals in states.items():
        novice = np.array(list(vals['novice'].values()))
        expert = np.array(list(vals['expert'].values()))
        A = np.vstack([novice, np.ones_like(novice)]).T
        a, b = np.linalg.lstsq(A, expert, rcond=None)[0]
        resid = np.linalg.norm(expert - (a * novice + b))
        print(state, round(a, 3), round(b, 3), round(resid, 4))

if __name__ == '__main__':
    main()
