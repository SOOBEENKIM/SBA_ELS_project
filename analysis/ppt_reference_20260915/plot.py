"""All presentation-style plots from saved prices; no MC calculations here."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';FIG=HERE/'figures'
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11,'axes.titleweight':'bold'})

def export(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        dest=FIG/f'{name}.{ext}';fig.savefig(dest,dpi=220,bbox_inches='tight',facecolor='white')
        if ext=='svg':dest.write_text('\n'.join(line.rstrip() for line in dest.read_text().splitlines())+'\n')
    plt.close(fig)

def metrics(d,cols):
    result={}
    for label,c in cols.items():
        e=(d.fair-d[c])*10000
        result[label]=dict(n=len(d),mean=float(e.mean()),median=float(e.median()),mae=float(e.abs().mean()),
                           rmse=float(np.sqrt((e**2).mean())),mape=float((e.abs()/(d.fair*10000)*100).mean()))
    return result

def plots(d,cols,prefix,caption):
    groups=pd.qcut(d.fair,6);g=d.assign(segment=groups).groupby('segment',observed=True)
    table=g.agg(n=('item','size'),fair=('fair','mean'))
    for label,c in cols.items():
        x=d.assign(segment=groups,e=(d.fair-d[c])*10000,a=(d.fair-d[c]).abs()/d.fair.abs()*100)
        ag=x.groupby('segment',observed=True).agg(gap=('e','mean'),mape=('a','mean'))
        table[label+'_gap']=ag.gap;table[label+'_mape']=ag.mape
    table.to_csv(OUT/f'{prefix}_segments.csv');xs=np.arange(6);width=.8/len(cols)
    fig,ax=plt.subplots(1,2,figsize=(13.5,4.8));colors=['#e07a5f','#1f6f9c','#2a9d8f']
    for j,label in enumerate(cols):
        pos=xs+(j-(len(cols)-1)/2)*width
        for a,key in zip(ax,['gap','mape']):a.bar(pos,table[label+'_'+key],width,label=label,color=colors[j])
    for a in ax:
        a.set_xticks(xs,[f'{i.left:.3f}–{i.right:.3f}' for i in table.index],rotation=30,ha='right');a.grid(axis='y',alpha=.2);a.set_axisbelow(True);a.legend(fontsize=9);a.set_xlabel('FAIR ratio segment (low → high)')
    ax[0].axhline(0,color='#666',lw=.8);ax[0].set_ylabel('mean (fair − MC) × 10,000');ax[0].set_title('Pricing difference by price segment')
    ax[1].set_ylabel('MAPE (%)');ax[1].set_title('|fair − MC| / fair by price segment')
    fig.suptitle(caption,fontsize=12);export(fig,prefix+'_segments')
    fig,a=plt.subplots(figsize=(12.5,4.8))
    for j,(label,c) in enumerate(cols.items()):
        e=(d.fair-d[c])*10000;a.hist(e,bins=90 if len(cols)==1 else 100,alpha=.75 if len(cols)==1 else .5,color=colors[j],label=f'{label}: mean {e.mean():,.2f}',edgecolor='white',lw=.2)
        a.axvline(e.mean(),color=colors[j],lw=1.6)
        if len(cols)==1:a.axvline(e.median(),color='#c0392b',lw=1.2,ls='--',label=f'median {e.median():,.2f}')
    a.axvline(0,color='#555',ls=':',lw=1);a.set_xlabel('(saved FAIR ratio − unit-face MC ratio) × 10,000');a.set_ylabel('Number of products');a.set_title(caption);a.legend();a.grid(axis='y',alpha=.2)
    export(fig,prefix+'_histogram');return metrics(d,cols)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');args=p.parse_args();FIG.mkdir(exist_ok=True)
    if args.fresh:
        d=pd.read_csv(OUT/'population_recomputed.csv');report={}
        report['iv']=plots(d,{'IV + source discount':'iv_boot'},'recomputed_iv',f'IV MC, {len(d):,} products; 40,000 paths')
        report['hv_iv']=plots(d,{'HV + NS':'hv_ns','IV + source discount':'iv_boot'},'recomputed_hv_iv',f'Same source settings, {len(d):,} products; 40,000 paths')
        report['iv_discount']=plots(d,{'IV + NS':'iv_ns','IV + source discount':'iv_boot'},'recomputed_iv_discount',f'Discount-method control, {len(d):,} products; 40,000 paths')
        name='recomputed_plot_summary.json'
    else:
        old=pd.read_csv(OUT/'reference_legacy_prices.csv');d=pd.read_csv(OUT/'reference_population_prices.csv')
        report={'legacy':plots(old,{'Archived HV':'mc'},'reference_legacy',f'Archived 23,151-product figure: {len(old):,} products'),
                'hv_iv':plots(d,{'Archived HV':'mc','Archived IV':'mc_iv'},'reference_hv_iv',f'Archived IV-comparison figure: {len(d):,} products')}
        name='reference_plot_summary.json'
    (OUT/name).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
