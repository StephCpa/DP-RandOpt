"""Assemble cross-setting evidence from archived audits without model inference."""
import json
from pathlib import Path


def main():
    root=Path('results/gpu-smoke')
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    original=load('completed_route1_audit/audit.json')['results']
    decomp=load('selection_decomposition/audit.json')['results']
    controls=load('batched_controls_20261023/analysis.json')
    gsm=load('gsm8k_pool_audit_20261025/analysis.json')
    rows=[]
    for filename,setting,seed,cap,N in [
        ('server_3b_n1000.npz','Countdown base-model legacy',20261005,128,1001),
        ('server_3b_n1000_seed20261008.npz','Countdown base-model legacy',20261008,128,1001),
        ('lzn_route1_n550.npz','Countdown Instruct nested prefix',20261013,512,550),
        ('lzn_route1_n1001_max512.npz','Countdown Instruct full pool',20261013,512,1001)]:
        x=next(r for r in original if r['file']==filename)
        rows.append(dict(setting=setting,seed=seed,N=N,cap=cap,base=x['base'],public=x['public'],nonprivate=x['nonprivate'],D1=x['D1']['mean'],
            random5=decomp['full_512']['random5']['vote'] if N==1001 and cap==512 else None,self_consistency=None,
            D1_minus_public=x['D1_minus_public'],D1_minus_nonprivate=x['D1_minus_nonprivate'],
            source='completed_route1_audit/audit.json',source_selector=filename))
    for cap in [512,1024,2048]:
        x=decomp[f'subset_{cap}']
        rows.append(dict(setting='Countdown Instruct first 32',seed=20261013,N=32,cap=cap,base=x['base'],
            **{k:x[k]['vote'] for k in ['public','nonprivate','D1','random5']},
            self_consistency=controls['self_consistency']['vote'] if cap==1024 else None,
            D1_minus_random5=x['D1']['vote_minus_random5'],source='selection_decomposition/audit.json',source_selector=f'subset_{cap}'))
    rows.append(dict(setting='GSM8K Instruct full pool',seed=20261024,item_seed=20261023,N=1001,cap=1024,base=gsm['base'],
        **{k:gsm[k]['vote'] for k in ['public','nonprivate','D1','random5']},self_consistency=gsm['standalone_baselines']['self_consistency']['vote'],
        comparisons=gsm['comparisons'],source='gsm8k_pool_audit_20261025/analysis.json'))
    out=root/'cross_task_synthesis_20261025';out.mkdir(exist_ok=True)
    data=dict(K=5,n=1000,Q=50,epsilon=8,delta=1e-5,rows=rows,
        scope='Different configurations are distinct settings, not pooled replications. Nested candidate prefixes share items/candidates. Null random5 entries avoid mixing older base-inclusive sampling with the later perturbation-only comparator.',
        mc_note='Full Countdown row uses the 400-draw completed-route audit D1=0.57159; the independent voting-decomposition audit has D1=0.57172. Their estimates are not silently interchanged.')
    (out/'evidence.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['| Setting / candidate seed | N | Cap | Base | Random-5 | Public | Nonprivate | D1 | Self-consistency |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{100*x:.2f}%'
    for r in rows:
        lines.append(f"| {r['setting']} / {r['seed']} | {r['N']} | {r['cap']} | "+' | '.join(fmt(r[k]) for k in ['base','random5','public','nonprivate','D1','self_consistency'])+' |')
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Assembled eight distinct settings; no new generations or mechanism draws.')


if __name__=='__main__':main()
