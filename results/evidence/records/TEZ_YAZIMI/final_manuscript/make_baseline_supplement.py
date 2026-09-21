"""Derive supplementary tables and one vector plot from frozen baseline counts."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M=Path(__file__).resolve().parent; ROOT=M.parents[1]
C=ROOT/'runs/operational/early_warning_multiseed_v1'
B=C/'single_model_baseline_v1'; P=B/'summary.json'
D=json.loads(P.read_text()); H=json.loads((C/'final_ten_model_seeds_v1/final_results.json').read_text())
F=['random','replay','drift','noise','targeted']; K=F+['family_macro_f1','pooled_f1','clean_fpr']
names={'modena':'Modena','ltown':'L-Town'}
labels={f:f.title()+' F1' for f in F}; labels.update(family_macro_f1='Family macro F1',pooled_f1='Pooled F1',clean_fpr=r'Clean FPR (\%)')
fmt=lambda k,v:f'{v*(100 if k=="clean_fpr" else 1):.4f}'
means={n:dict(H['networks'][n]['mean']) for n in names}
for n in means: means[n]['family_macro_f1']=statistics.mean(means[n][f] for f in F)
parts=[]
for n,title in names.items():
    arms=D['networks'][n]
    for arm,r in arms.items():
        assert len(r['rows'])==60
        for k in K: assert abs(statistics.mean(row[k] for row in r['rows'])-r['mean'][k])<1e-12
    parts += [r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption[Single-classifier comparison on '+title+r']{'+title+r': single-classifier controls versus the frozen hybrid, each averaged over ten model seeds and six shared sources. Single-classifier thresholds maximize calibration pooled F1 under the common clean-FPR ceiling; the hybrid retains its recorded operating points. Clean FPR is a percentage.}',
        r'\label{tab:single'+n+r'}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
        r'Metric & Single, 0 h & Single, 3 h & Hybrid\\',r'\midrule']
    for k in K:
        parts.append(' & '.join([labels[k],fmt(k,arms['delta0_pooled']['mean'][k]),fmt(k,arms['delta3_pooled']['mean'][k]),fmt(k,means[n][k])])+r'\\')
    parts += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    parts += [r'\FloatBarrier', r'\input{tables/single_model_'+n+r'_interpretation}']
parts += [r'\figfile{single_model_paired.pdf}{1.0}{Paired hybrid effects relative to the three-hour single classifier}{Family-F1 differences between the frozen hybrid and the three-hour single classifier at its pooled-F1 operating point. Each circle averages ten model seeds within one evaluation source; diamonds mark the six-source mean. Positive values favor the hybrid. The source spread is descriptive, not a confidence interval.}{fig:singlepaired}',r'\FloatBarrier',r'\input{tables/single_model_paired_interpretation}']
(M/'tables/single_model_results.tex').write_text('\n'.join(parts)+'\n')

appendix=[r'\section{Supplementary single-classifier evidence}',r'\label{sec:baselineappendix}',
    r'The supplementary comparison retains both calibration objectives for every fitted classifier. Table~\ref{tab:balancedbaseline} reports the alternative thresholds selected to maximize worst-family F1 under the same clean-FPR ceiling. These points were frozen alongside the pooled-F1 points before evaluation; neither objective was chosen using the reported evaluation scores. The full per-source confusion counts, per-model settings, input hashes and frozen rules are supplied under \texttt{single\_model\_baseline\_v1/} in the review package \cite{projectbaseline}.',
    r'\begin{table}[!htbp]',r'\centering\small',r'\caption[Single-classifier results with family-balanced calibration]{Single-classifier means when calibration maximizes worst-family F1. Both time allowances use ten model seeds and six sources per network. Clean FPR is a percentage.}',
    r'\label{tab:balancedbaseline}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
    r'& \multicolumn{2}{c}{Modena} & \multicolumn{2}{c}{L-Town}\\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Metric & Single, 0 h & Single, 3 h & Single, 0 h & Single, 3 h\\',r'\midrule']
for k in K:
    appendix.append(' & '.join([labels[k]]+[fmt(k,D['networks'][n][f'delta{d}_balanced']['mean'][k]) for n in names for d in [0,3]])+r'\\')
appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\FloatBarrier',
    r'The historical latency comparison in Table~\ref{tab:historicaldelay} is supplied separately as \texttt{latency\_deployment\_v4/eval\_report.json}, its frozen selection and the evaluation script. That record aggregates one earlier configuration across six sources. The supplement verifier recomputes family F1 from its counts and checks the link to the frozen selection. It does not treat these historical scores as final ten-seed results.']
(M/'tables/single_model_appendix.tex').write_text('\n'.join(appendix)+'\n')

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7,3.4),sharey=True,sharex=True)
paired={}
for ax,(n,title) in zip(axes,names.items()):
    base=D['networks'][n]['delta3_pooled']['by_source']; hybrid=H['networks'][n]['by_source_seed']
    values=np.array([[hybrid[s][f]-base[s][f] for s in base] for f in F])
    paired[n]={f:values[j].tolist() for j,f in enumerate(F)}
    for j in range(5):
        ax.scatter(values[j],j+np.linspace(-.12,.12,6),s=21,facecolors='none',edgecolors='#52788e',linewidths=.9)
    ax.scatter(values.mean(axis=1),np.arange(5),s=31,marker='D',color='#202a35',zorder=3)
    ax.axvline(0,color='#777777',linestyle='--',linewidth=.8)
    ax.set_title(title); ax.set_yticks(np.arange(5),[f.title() for f in F]); ax.set_xlabel('Hybrid minus single-classifier F1')
    ax.grid(axis='x',color='#dddddd',linewidth=.5); ax.set_axisbelow(True)
axes[0].invert_yaxis(); fig.tight_layout(w_pad=1.8)
fig.savefig(M/'figures/single_model_paired.pdf',bbox_inches='tight');plt.close(fig)
audit={'summary_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
    'final_results_sha256':hashlib.sha256((C/'final_ten_model_seeds_v1/final_results.json').read_bytes()).hexdigest(),
    'source_paired_family_differences':paired,'means_verified':True,'new_plot_uses_only_stored_counts':True}
(M/'audit/baseline_tables.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'tables':3,'figures':1,'means_verified':True},indent=2))
