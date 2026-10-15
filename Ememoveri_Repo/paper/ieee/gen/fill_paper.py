import json, re
root = 'results/real_ext'
R, B, E, X, V05 = (json.load(open(f'{root}/{n}.json')) for n in ('summary_v06', 'summary_v06b', 'ev2_dev', 'summary_ext', 'summary_v05'))
F, H, ML, DT = R['fresh16'], R['heldout12'], R['multilang8'], R['dev_test']
m3 = lambda x: ('$-$' if x < -0.0005 else '') + f'{abs(x):.3f}' if abs(x) >= 0.0005 else '0.000'
ci = lambda t: f'{m3(t[0])} [{m3(t[1])}, {m3(t[2])}]'
pv = lambda p: '<0.001' if p < 0.001 else f'={p:.3f}'
def con(S, k, met='hit5'):
    c = S['contrasts'][k][met]
    return c
V = {}
for tag, S in (('F', F), ('H', H), ('M', ML)):
    for key, nm in (('H1', 'H1 EV2 - Snapshot'), ('H2', 'H2 EV2 - EmemoVeri-T'), ('H3', 'H3 EV2 - EV2 plain'), ('H4', 'H4 EV2 - Sim+State+Rec+Pop'),
                    ('H5a', 'H5a EV2 - all channels (linear)'), ('H5b', 'H5b EV2 - learned ranker (GBT)'), ('SH', 'EV2 - similar-history (v0.5)'), ('D', 'EV2 - EmemoVeri-D'), ('LG', 'EV2 - learned ranker (logistic)')):
        c = con(S, nm)
        V[f'{tag}{key}'] = ci(c['macro']); V[f'{tag}{key}wl'] = f"{c['wins']} of {len(S['repos'])}" if c['losses'] == 0 else f"{c['wins']} wins and {c['losses']} losses"
        V[f'{tag}{key}p'] = pv(c['p_signflip']); V[f'{tag}{key}holm'] = pv(c['p_holm']) if 'p_holm' in c else '--'; V[f'{tag}{key}dz'] = f"{c['dz']:.2f}"
        V[f'{tag}{key}pt'] = m3(c['macro'][0])
    V[f'{tag}n'] = f"{S['n_total']:,}".replace(',', '{,}')
V['Fpy10'] = {k: ci(F['python10']['contrasts'][nm]['hit5']['macro']) for k, nm in (('H1', 'H1 EV2 - Snapshot'), ('H2', 'H2 EV2 - EmemoVeri-T'), ('H3', 'H3 EV2 - EV2 plain'))}
V['Foth6'] = {k: ci(F['other6']['contrasts'][nm]['hit5']['macro']) for k, nm in (('H1', 'H1 EV2 - Snapshot'), ('H2', 'H2 EV2 - EmemoVeri-T'), ('H3', 'H3 EV2 - EV2 plain'))}
V['Foth6p'] = pv(F['other6']['contrasts']['H1 EV2 - Snapshot']['hit5']['p_signflip'])
nv = F['novel']['contrasts']
V['Fnov1'] = ci(nv['H1 EV2 - Snapshot']['hit5']['macro']); V['Fnov3'] = ci(nv['H3 EV2 - EV2 plain']['hit5']['macro']); V['Fnov4'] = ci(nv['H4 EV2 - Sim+State+Rec+Pop']['hit5']['macro'])
V['Fnovn'] = f"{F['novel']['n_total']:,}".replace(',', '{,}'); V['Fdropped'] = f"{F['n_total'] - F['novel']['n_total']}"
s = F['systems']
V['F_snap'], V['F_ev2'], V['F_plain'], V['F_T'], V['F_gbt'] = (f"{s[k]['macro_hit5']:.3f}" for k in ('Snapshot RAG', 'EmemoVeri-2', 'EmemoVeri-2 plain', 'EmemoVeri-T', 'Learned ranker (GBT)'))
V['Fbody_s'], V['Fbody_b'] = F['body']['subject'], F['body']['subject_plus_body']
V['Fbodyn'] = f"{F['body']['n']:,}".replace(',', '{,}')
V['dT_abl'] = ci(E['test']['aware_minus_plain']['task']); V['dT_file'] = ci(E['test']['aware_minus_plain']['file'])
V['dev_ev2'], V['dev_plain'] = f"{E['test']['aware']['hit5']:.3f}", f"{E['test']['plain']['hit5']:.3f}"
D2 = R['dev_test']['ev2_minus']
V['dev_vs_snap'], V['dev_vs_T'], V['dev_vs_ssrp'], V['dev_vs_gbt'] = ci(D2['Snapshot RAG']['task']), ci(D2['EmemoVeri-T']['task']), ci(D2['Sim+State+Rec+Pop']['task']), ci(D2['Learned ranker (GBT)']['task'])
bd = B['degrade']
V['deg'] = {l: {k: bd[l][k]['hit5'] for k in ('Snapshot RAG', 'EmemoVeri-T', 'Sim+State+Rec+Pop', 'EmemoVeri-2 plain', 'EmemoVeri-2')} for l in bd}
V['deg_gain'] = {l: ci(bd[l]['Snapshot RAG']['ev2_minus']) for l in bd}
V['deg_gainT'] = {l: ci(bd[l]['EmemoVeri-T']['ev2_minus']) for l in bd}
rv = B['revert']
V['rv_cancelled'] = sum(v['cancelled_commits'] for v in rv.values()); V['rv_changed'] = sum(v['ranking_changed'] for v in rv.values())
V['rv_tasks'] = f"{sum(v['n_tasks'] for v in rv.values()):,}".replace(',', '{,}')
dev_body = R['dev_test_body']


HELD = json.load(open(f'{root}/summary_heldout.json'))
pc = lambda x: f'{100 * x:.1f}\\%'
def life(L):
    d, r = L['delete'], L['rename']
    return {'d_n': str(d['n']), 'r_n': str(r['n']), 'd_naive': pc(d['post_old_in_top5_naive']), 'd_pre': pc(d['pre_hit_naive']),
            'r_new_carry': pc(r['post_new_in_top5_carry:snapshot']), 'r_new_nocarry': pc(r['post_new_in_top5_nocarry:snapshot']),
            'r_cont_carry': pc(r['continuity_given_pre_hit_carry']), 'r_cont_nocarry': pc(r['continuity_given_pre_hit_nocarry']), 'r_pre_n': str(r['n_pre_hit_snapshot'])}
V['L'] = {'dev': life(X['lifecycle']), 'held': life(HELD['lifecycle']), 'ml': life(V05['multilang8_lifecycle'])}
for tag, S in (('H', V05['heldout12']), ('M', V05['multilang8'])):
    for key, nm in (('o1', 'H1 EmemoVeri-T - Snapshot RAG'), ('o2', 'H2 EmemoVeri-T - Sim+State+Rec+Pop'), ('o3', 'H3 EmemoVeri-D - Snapshot RAG'), ('o4', 'H4 EmemoVeri-T - similar-history (tuned)')):
        c = S['contrasts'][nm]['hit5']
        V[f'{tag}{key}'] = ci(c['macro']); V[f'{tag}{key}wl'] = f"{c['wins']}/{c['losses']}"; V[f'{tag}{key}holm'] = pv(c['p_holm']); V[f'{tag}{key}dz'] = f"{c['dz']:.2f}"


fb = lambda d, k: f"{d[k]:.3f}"
V['FB'] = {'ss': fb(F['body']['subject'], 'Snapshot RAG'), 'sp': fb(F['body']['subject'], 'EmemoVeri-2 plain'), 'se': fb(F['body']['subject'], 'EmemoVeri-2'),
           'bs': fb(F['body']['subject_plus_body'], 'Snapshot RAG'), 'bp': fb(F['body']['subject_plus_body'], 'EmemoVeri-2 plain'), 'be': fb(F['body']['subject_plus_body'], 'EmemoVeri-2')}
V['dg'] = {l.replace('.', '_'): {'ev2': f"{bd[l]['EmemoVeri-2']['hit5']:.3f}", 'snap': f"{bd[l]['Snapshot RAG']['hit5']:.3f}"} for l in bd}
V['dg_gain'] = {l.replace('.', '_'): ci(bd[l]['Snapshot RAG']['ev2_minus']) for l in bd}
for tag, S in (('M', ML),):
    c = S['contrasts']['H2 EV2 - EmemoVeri-T']['hit5']; V['MH2'] = ci(c['macro'])

AB = json.load(open(f'{root}/ablate_ev2.json')); SE = json.load(open(f'{root}/sens_ev2.json'))
def abv(tag, name):
    f = AB[tag]['variants'][name]['full_minus']; return f
V['AB'] = {}
for tg, key in (('fresh', 'fresh16'), ('heldout', 'heldout12')):
    d = {}
    for k, nm in (('ev', '- evidence-item term'), ('state', '- state filter'), ('sim', '- claim similarity'), ('rec', '- recency prior'), ('str', '- strength weighting'),
                  ('freq', '- frequency prior'), ('both', '- both priors'), ('carry', '- rename carry-over')):
        if nm in AB[key]['variants'] and 'full_minus' in AB[key]['variants'][nm]:
            f = AB[key]['variants'][nm]['full_minus']; d[k] = ci(f['macro']); d[k + 'holm'] = pv(f['p_holm'])
    V['AB'][tg] = d
V['AB']['fresh']['ev'] = ci(abv('fresh16', '- evidence-item term')['macro'])
share = lambda x: f"{int(round(100 * x))}\\% of {SE['n']}"
V['S'] = {'n': str(SE['n']), 'min': f"{SE['min']:.3f}", 'max': f"{SE['max']:.3f}", 'median': f"{SE['median']:.3f}", 'rank': str(SE['frozen_rank']),
          'snap': f"{SE['reference']['Snapshot RAG']:.3f}", 'evt': f"{SE['reference']['EmemoVeri-T']:.3f}", 'ssrp': f"{SE['reference']['Sim+State+Rec+Pop']:.3f}"}

for tag, S in (('H', H), ('M', ML)):
    for key, nm in (('p1', 'H1 EV2 - Snapshot'), ('p2', 'H2 EV2 - EmemoVeri-T'), ('p3', 'H3 EV2 - EV2 plain'), ('p4', 'H4 EV2 - Sim+State+Rec+Pop')):
        c = S['contrasts'][nm]['hit5']; V[f'{tag}{key}'] = ci(c['macro']); V[f'{tag}{key}wl'] = f"{c['wins']}/{c['losses']}"

body = open('paper/ieee/gen/paper_template.tex').read()
def sub(m):
    key = m.group(1)
    v = V
    for part in key.split('.'):
        v = v[part]
    return str(v)
body = re.sub(r'<<([A-Za-z0-9_.]+)>>', sub, body)
open('paper/ieee/ememoveri_ieee.tex', 'w').write(body)
print(len(body.split()))
