"""Generate tables, figures and the report exclusively from completed saved results."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';FIG=HERE/'figures'
sys.path[:0]=[str(HERE.parents[1]),str(HERE.parents[1]/'analysis/mc_monthly_v3_20260910')]
from evaluate import metric
AXIS={'coupon_regular':'Annual coupon','ki_barrier':'KI barrier','first_strike':'First strike','last_strike':'Final strike','monthly_barrier':'Monthly coupon barrier','nobs_early':'Observation +1: early','nobs_middle':'Observation +1: middle','nobs_late':'Observation +1: late','tenor_months':'Tenor (months)'}
KO={'coupon_regular':'연 쿠폰 +1%p','ki_barrier':'낙인 배리어 +5%p','first_strike':'1차 행사가 +5%p','last_strike':'만기 행사가 +5%p','monthly_barrier':'월 쿠폰 배리어 +5%p','nobs_early':'관측 +1회: 첫 평가 전','nobs_middle':'관측 +1회: 중간','nobs_late':'관측 +1회: 마지막 구간','tenor_months':'만기 변경'}
COL={'reference':'#e07a5f','detailed':'#1f6f9c'}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'axes.titleweight':'bold'})
def table(d):
    def fmt(x):
        if pd.isna(x):return '—'
        return f'{x:,.2f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    cols=list(d);return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
def export(fig,name):
    fig.tight_layout(pad=2,w_pad=3,h_pad=3)
    for ext in ['png','pdf','svg']:
        dest=FIG/f'{name}.{ext}';fig.savefig(dest,dpi=220,bbox_inches='tight',facecolor='white')
        if ext=='svg':dest.write_text('\n'.join(line.rstrip() for line in dest.read_text().splitlines())+'\n')
    plt.close(fig)
def load(name):return json.loads((OUT/name).read_text())

def synthetic_reports():
    assert load('synthetic_complete.json')['status']=='complete'
    labels=pd.read_csv(OUT/'synthetic_labels.csv.gz');checkpoints=pd.read_csv(OUT/'synthetic_checkpoints.csv.gz')
    effects=[]
    for keys,g in labels.groupby(['payoff','split','origin','monthly','structure','axis','offset','scope']):
        r=dict(zip(['payoff','split','origin','monthly','structure','axis','offset','scope'],keys))
        r.update(n_families=g.family.nunique(),mean_delta=g.delta_krw.mean(),median_delta=g.delta_krw.median(),min_delta=g.delta_krw.min(),max_delta=g.delta_krw.max(),mean_delta_se=g.delta_se_krw.mean(),positive=int((g.delta_krw>1e-8).sum()),negative=int((g.delta_krw< -1e-8).sum()))
        effects.append(r)
    pd.DataFrame(effects).to_csv(OUT/'mc_effects_by_structure.csv',index=False)
    targets={'coupon_regular':.01,'ki_barrier':.05,'first_strike':.05,'last_strike':.05,'monthly_barrier':.05,'nobs_early':1,'nobs_middle':1,'nobs_late':1}
    tests=labels[labels.split.eq('test')];summary=[]
    for (payoff,monthly,axis,offset),g in tests.groupby(['payoff','monthly','axis','offset']):
        if axis in targets and np.isclose(offset,targets[axis]) or axis=='tenor_months':
            summary.append(dict(payoff=payoff,monthly=monthly,condition=KO[axis]+(f' {offset:+g}개월' if axis=='tenor_months' else ''),axis=axis,offset=offset,n=g.family.nunique(),mean_delta_krw=g.delta_krw.mean(),min_delta_krw=g.delta_krw.min(),max_delta_krw=g.delta_krw.max()))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'mc_effect_summary.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    nobs_axes=['nobs_early','nobs_middle','nobs_late'];xx=np.arange(3)
    for ax,monthly in zip(axs,[False,True]):
        for j,arm in enumerate(['reference','detailed']):
            g=tests[tests.payoff.eq(arm)&tests.monthly.eq(monthly)&tests.axis.isin(nobs_axes)].groupby('axis').delta_krw.mean().reindex(nobs_axes)
            ax.bar(xx+(j-.5)*.36,g,.36,color=COL[arm],label=arm)
        ax.set_xticks(xx,['Early','Middle','Late']);ax.axhline(0,color='#777',lw=.8);ax.set_ylabel('Mean MC increment (KRW)');ax.set_title('Monthly coupons' if monthly else 'Regular coupons');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('One additional observation: source projection vs explicit schedule');export(fig,'mc_observation_count')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    impl=tests[tests.payoff.eq('detailed_minus_reference')]
    for ax,base in zip(axs,[True,False]):
        g=impl[impl.axis.eq('base')] if base else impl[impl.axis.ne('base')&impl.scope.eq('terms')]
        value=g.price_krw if base else g.delta_krw
        ax.hist(value,bins=60,color='#2a9d8f',alpha=.8);ax.axvline(0,color='#555',ls=':');ax.set_xlabel('Detailed minus reference (KRW)');ax.set_ylabel('Contract/scenario count');ax.set_title('Baseline price difference' if base else 'Increment difference')
    fig.suptitle('Paired payoff-implementation comparison on identical paths');export(fig,'payoff_implementation_difference')
    for axis in ['coupon_regular','ki_barrier','first_strike','last_strike','monthly_barrier','tenor_months']:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for ax,monthly in zip(axs,[False,True]):
            for arm in ['reference','detailed']:
                g=tests[tests.payoff.eq(arm)&tests.monthly.eq(monthly)&tests.axis.eq(axis)]
                if g.empty:continue
                z=g.groupby('offset').delta_krw.agg(['mean',lambda s:s.quantile(.1),lambda s:s.quantile(.9)])
                x=z.index.to_numpy()*(1 if axis=='tenor_months' else 100)
                ax.plot(x,z['mean'],marker='o',label=arm,color=COL[arm]);ax.fill_between(x,z.iloc[:,1],z.iloc[:,2],alpha=.10,color=COL[arm])
            ax.axhline(0,color='#777',lw=.8);ax.set_title('Monthly coupons' if monthly else 'Regular coupons');ax.set_ylabel('MC increment (KRW / 10,000 face)');ax.set_xlabel('Months' if axis=='tenor_months' else 'Change (percentage points)');ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        fig.suptitle(AXIS[axis]+' — mean and 10–90% contract range');export(fig,'mc_curve_'+axis)
    conv=checkpoints[checkpoints.paths.eq(20000)].merge(checkpoints[checkpoints.paths.eq(40000)],on=['payoff','family','axis','offset'],suffixes=('_20k','_40k'))
    conv['delta_change_krw']=conv.delta_krw_40k-conv.delta_krw_20k
    conv.to_csv(OUT/'mc_convergence.csv.gz',index=False)
    convsum=conv[conv.split_40k.eq('test')&conv.axis.ne('base')].groupby(['payoff','axis']).agg(n=('family','size'),mean_abs_change=('delta_change_krw',lambda s:s.abs().mean()),p95_abs_change=('delta_change_krw',lambda s:s.abs().quantile(.95)),mean_mc_se=('delta_se_krw_40k','mean')).reset_index()
    convsum.to_csv(OUT/'mc_convergence_summary.csv',index=False)
    seed=pd.read_csv(OUT/'synthetic_seed_checkpoints.csv.gz');seed=seed[seed.paths.eq(40000)&seed.split.eq('test')]
    seed['delta_krw']=seed.delta_sum/seed.paths*10000
    seed.groupby(['payoff','monthly','axis','offset','replicate']).delta_krw.mean().reset_index().to_csv(OUT/'mc_effect_by_seed.csv',index=False)
    return summary,convsum

def model_reports():
    metrics=pd.read_csv(OUT/'model_metrics.csv');pred=[];selections={}
    for payoff in ['reference','detailed']:
        base=HERE/'synthetic'/payoff
        decision=json.loads((base/'results/model_selection.json').read_text());selections[payoff]=decision['selected_arm']
        assert json.loads((base/'results/evaluation_complete.json').read_text())['status']=='complete'
        p=pd.read_csv(base/'results/predictions.csv.gz',dtype={'seed':str});pred.append(p)
    predictions=pd.concat(pred,ignore_index=True)
    chosen=pd.concat([predictions[predictions.payoff.eq(a)&predictions.model.eq(m)&predictions.seed.eq('ensemble')] for a,m in selections.items()])
    test=chosen[chosen.cohort.isin(['regular_test','monthly_test'])]
    for name,subset,keys in [
        ('model_baseline_metrics',predictions[predictions.axis.eq('base')],['payoff','cohort','model','seed']),
        ('model_metrics_by_structure',predictions,['payoff','cohort','model','seed','scope','structure']),
        ('model_selected_axis',chosen,['payoff','cohort','scope','axis']),
        ('model_seed_summary',predictions,['payoff','cohort','model','seed','scope'])]:
        rows=[dict(zip(keys,k),**metric(g)) for k,g in subset.groupby(keys)]
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    for scope in ['terms','schedule']:
        fig,axs=plt.subplots(2,2,figsize=(10,8))
        for row,cohort in enumerate(['regular_test','monthly_test']):
            for col,arm in enumerate(['reference','detailed']):
                ax=axs[row,col];g=test[test.scope.eq(scope)&test.axis.ne('base')&test.cohort.eq(cohort)&test.payoff.eq(arm)]
                ax.scatter(g.mc_delta_krw,g.pred_delta_krw,s=4,alpha=.2,color=COL[arm]);lim=[min(g.mc_delta_krw.min(),g.pred_delta_krw.min()),max(g.mc_delta_krw.max(),g.pred_delta_krw.max())]
                ax.plot(lim,lim,'--',color='#555',lw=1);ax.set_title(f'{arm} / {cohort}');ax.set_xlabel('MC increment (KRW)');ax.set_ylabel('DeepONet increment (KRW)');ax.grid(alpha=.15)
        fig.suptitle(f'Frozen selected models: {scope} increments');export(fig,'model_increment_'+scope)
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for row,cohort in enumerate(['regular_test','monthly_test']):
        for col,arm in enumerate(['reference','detailed']):
            ax=axs[row,col];g=test[test.scope.eq('terms')&test.cohort.eq(cohort)&test.payoff.eq(arm)]
            ax.scatter(g.mc_price_krw,g.pred_price_krw,s=4,alpha=.2,color=COL[arm]);lo=min(g.mc_price_krw.min(),g.pred_price_krw.min());hi=max(g.mc_price_krw.max(),g.pred_price_krw.max())
            ax.plot([lo,hi],[lo,hi],'--',color='#555',lw=1);ax.set_title(f'{arm} / {cohort}');ax.set_xlabel('MC price (KRW)');ax.set_ylabel('DeepONet price (KRW)')
    fig.suptitle('Price predictions on the same held-out synthetic contracts');export(fig,'model_prices')
    resolved=(chosen.mc_delta_krw.abs()>1.96*chosen.mc_delta_se_krw)&chosen.affected.ge(30)
    failure=chosen[chosen.axis.ne('base')&resolved&(np.sign(chosen.mc_delta_krw)!=np.sign(chosen.pred_delta_krw))].copy()
    failure['abs_delta_error']=failure.delta_error_krw.abs();failure.sort_values('abs_delta_error',ascending=False).to_csv(OUT/'resolved_direction_failures.csv.gz',index=False)
    summary=metrics[metrics.seed.astype(str).eq('ensemble')&metrics.axis.eq('ALL')&metrics.cohort.isin(['regular_test','monthly_test','published_monthly_6'])]
    summary.to_csv(OUT/'model_summary.csv',index=False)
    # Within-label, family-clustered paired improvement; no cross-payoff causal claim.
    boots=[];rng=np.random.default_rng(2026091517)
    for payoff,selected in selections.items():
        for cohort in ['regular_test','monthly_test']:
            g=predictions[predictions.payoff.eq(payoff)&predictions.cohort.eq(cohort)&predictions.seed.eq('ensemble')&predictions.scope.eq('terms')&predictions.axis.ne('base')]
            for comparator in ['augmented_price','augmented_delta','affine_coupon_delta','prior_frozen_affine_coupon_delta']:
                if comparator==selected:continue
                a=g[g.model.eq(selected)].groupby('family').delta_error_krw.apply(lambda s:s.abs().mean());b=g[g.model.eq(comparator)].groupby('family').delta_error_krw.apply(lambda s:s.abs().mean());diff=(a-b).dropna().to_numpy()
                sample=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
                boots.append(dict(payoff=payoff,cohort=cohort,selected=selected,comparator=comparator,mean_difference=diff.mean(),ci_low=np.quantile(sample,.025),ci_high=np.quantile(sample,.975)))
    pd.DataFrame(boots).to_csv(OUT/'paired_model_improvement.csv',index=False)
    dispersion=predictions[predictions.seed.ne('ensemble')&predictions.axis.ne('base')].groupby(['payoff','model','family','axis','offset']).pred_delta_krw.agg(['mean','std','min','max']).reset_index()
    dispersion.to_csv(OUT/'training_seed_dispersion.csv.gz',index=False)
    return summary,selections,len(failure)

def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic-only',action='store_true');a=p.parse_args();FIG.mkdir(exist_ok=True)
    effects,conv=synthetic_reports();model=None
    if (OUT/'model_metrics.csv').exists():model=model_reports()
    lines=['# 발표 기준 Pricer 재현 및 두 지급 구현의 증분 평가','',
      '실제 그림을 만든 코드·저장 표본을 먼저 고정하고 제공된 ATM IV를 연결했다. 같은 조건의 4만 경로 재계산과, 같은 합성 계약에 대한 두 지급 구현의 MC·DeepONet 평가를 구분한다. 기존 main과 이전 IV branch의 결과는 보존했다.','',
      '## 그림의 출처 확인','',
      '앞선 23,151개 STEP KI 그림과 이후 58,790개 4구조 IV 비교 그림은 서로 다른 저장 표본이다. 다음은 기존 저장 가격을 재집계한 값이며 새 4만 경로 결과와 구분한다.','']
    rp=load('reference_plot_summary.json');rr=[]
    for cohort,arms in rp.items():
        for arm,r in arms.items():rr.append(dict(표본=cohort,가격=arm,상품수=r['n'],평균차이=r['mean'],중앙값=r['median'],MAE=r['mae'],MAPE=r['mape']))
    lines += [table(pd.DataFrame(rr)),'![원본 표본의 HV·IV 구간별 차이](figures/reference_hv_iv_segments.png)','',
      '## 고정한 계산 조건','',
      '| 항목 | 재현 기준 |\n|---|---|\n| 상품 | 원본 58,790개 ID·계약 조건 그대로 |\n| HV·상관 | 발행일 전 180개 수익률, 최소 60개; HV는 표본표준편차×√252 |\n| IV | 원본 8개 지수 매핑, 발행일 이하 최근 ATM IV; 미제공 자산은 HV |\n| 할인 | HV는 NS; 원본 IV 그림은 3개 금리점의 선형 제로금리 보간 |\n| 추가 대조 | IV를 유지하고 NS 할인으로 계산 |\n| 경로 | 일별 GBM, q=0, dt=1/365, 원본 평가일 반올림과 난수 시드 |\n| 실제 상품 MC | 40,000경로·상품별 1시드; 원본 IV 100,000경로와 구분 |\n| 합성 MC | 학습·검증 40,000경로×1시드, 시험·진단 40,000경로×3시드 |','',
      '원본 코드의 `bootstrap`은 여기서는 3개 금리를 제로금리로 간주한 선형 보간이다. 채권 현금흐름을 순차적으로 부트스트래핑한 구현은 아니다. 이 명칭과 구현을 구분했다.','',
      '원본 plot 계산식을 그대로 재현했다: `(fair - mc) × 10,000`, `abs(fair - mc) / abs(fair) × 100`. `fair`는 발행가로 나눈 관측값이고 `mc`는 단위 액면 현금흐름의 현재가치이다. 발행가와 액면가가 다른 상품에서 두 분모가 일치한다고 주장하지 않는다. 합성 가격·증분과 모델 오차는 모두 액면 10,000원 기준이다.','']
    if (OUT/'population_complete.json').exists():
        new=load('recomputed_plot_summary.json');rr=[]
        for arm,r in new['hv_iv'].items():rr.append(dict(가격=arm,상품수=r['n'],평균차이=r['mean'],중앙값=r['median'],MAE=r['mae'],MAPE=r['mape']))
        r=new['iv_discount']['IV + NS'];rr.insert(1,dict(가격='IV + NS',상품수=r['n'],평균차이=r['mean'],중앙값=r['median'],MAE=r['mae'],MAPE=r['mape']))
        lines += ['## 4만 경로 재계산','',table(pd.DataFrame(rr)),
          '![4만 경로 HV·IV 구간별 차이](figures/recomputed_hv_iv_segments.png)','![4만 경로 HV·IV 차이 분포](figures/recomputed_hv_iv_histogram.png)','![IV만 사용한 구간별 차이](figures/recomputed_iv_segments.png)','![IV 할인곡선 비교](figures/recomputed_iv_discount_segments.png)','']
    else:lines+=['## 실제 상품 전량 재계산','', '**진행 중.** 아래 합성 결과와 실제 상품 전량 완료 여부를 구분한다.','']
    lines += ['## 두 지급 구현과 합성 계약','',
      '- `reference`: 원본의 균등 평가일·평가일 할인·미낙인 만기 원금 지급·월지급형 누적 쿠폰 근사.\n- `detailed`: 기존 보완 구현의 명시적 평가/지급일·만기 생존 쿠폰·월 쿠폰 현금흐름.\n- 두 구현에 같은 기준·변경 계약, 같은 IV·상관·금리, 같은 경로를 사용했다. 구현 차이 자체의 paired MC 표준오차도 계산했다.\n- 기준 계약 2,652개와 88,181개 시나리오. 기존 2,048/256/256 학습·검증·시험 계약 분할 및 92개 진단 계약을 유지했다. 계약 템플릿 분할은 기존 것을 유지했으며 시장 상태까지 독립적으로 분할한 실험은 아니다.\n- 쿠폰·KI·1차/만기 행사가·월 쿠폰 배리어를 여러 양/음 변경 폭으로 평가했다. 관측 추가와 만기 변경은 기존 명시된 일정 규칙을 적용한 평가 축이며 학습에는 넣지 않았다.','',
      '**원본 근사에서 월 쿠폰 배리어와 개별 지급일은 가격 입력으로 사용되지 않는다.** 따라서 이 축의 원본 증분이 0인 것은 모델이 월 쿠폰을 정확하게 구현했다는 증거가 아니다. 관측 횟수가 바뀌면 원본은 모든 평가일을 다시 균등 배치하므로 상세 구현의 평가일 한 개 삽입과 표현 방식도 다르다.','',
      '## 조건별 평균 MC 증분','']
    compact=effects[effects.payoff.isin(['reference','detailed'])].copy();compact['상품']=np.where(compact.monthly,'월지급형','일반형')
    compare=compact.pivot(index=['monthly','condition','axis','offset','n'],columns='payoff',values='mean_delta_krw').reset_index()
    compare['차이: 상세−원본']=compare.detailed-compare.reference;compare['상품']=np.where(compare.monthly,'월지급형','일반형')
    compare=compare.rename(columns={'condition':'변경','n':'계약수','reference':'원본 근사 ΔMC','detailed':'상세 지급 ΔMC'})
    compare.to_csv(OUT/'payoff_effect_comparison.csv',index=False)
    schedule_axes=['nobs_early','nobs_middle','nobs_late','tenor_months']
    columns=['상품','변경','계약수','원본 근사 ΔMC','상세 지급 ΔMC','차이: 상세−원본']
    lines += ['일반형·월지급형 시험 계약 각각 128개에서 계산한 평균이며 단위는 원이다. KI 변경은 KI 계약만 포함한다. 아래 구현 간 차이는 **상세 구현의 증분 − 원본 근사의 증분**이다. 계약별 최소·최대와 구조별 결과는 CSV에 별도로 저장했다.','',
      table(compare[~compare.axis.isin(schedule_axes)][columns]),
      '### 관측 횟수·만기 변경 규칙과 결과','',
      '관측 +1회는 첫 평가 전·중간 구간·마지막 구간의 중간 날짜에 추가한다. 중간·마지막의 추가 행사가와 쿠폰 누적기간은 양옆 조건을 선형 보간하고, 첫 평가 전은 첫 행사가를 유지한다. 추가 Lizard 조건은 두지 않는다. 상세 구현은 기존 날짜를 유지하지만 원본 근사에서는 전체 관측 날짜를 다시 균등 배치한다. 입력 슬롯이 12개인 계약은 추가 관측 실험에서 제외되어 일반형은 81개다.','',
      '만기는 ±1/3/6/12개월 변경한다. 조기상환 평가일과 일반형 쿠폰 누적기간은 새 만기 비율로 조정한다. 월지급형은 기존 월별 지급을 유지하고, 연장 시 월 지급을 추가하며 단축 시 만기 이후 지급을 제거하고 마지막 짧은 기간의 쿠폰을 비례 계산한다. 월지급형은 저장된 영업일·지급 지연 규칙을 적용하고 일반 합성 계약은 달력 일수 규칙을 쓴다. 따라서 이 두 축은 부수 일정까지 명시해 변경한 계약의 효과다.','',
      table(compare[compare.axis.isin(schedule_axes)][columns]),
      '![쿠폰 민감도](figures/mc_curve_coupon_regular.png)','![낙인 배리어 민감도](figures/mc_curve_ki_barrier.png)','![1차 행사가 민감도](figures/mc_curve_first_strike.png)','![만기 행사가 민감도](figures/mc_curve_last_strike.png)','![월 쿠폰 배리어 민감도](figures/mc_curve_monthly_barrier.png)','![관측 횟수 민감도](figures/mc_observation_count.png)','![만기 민감도](figures/mc_curve_tenor_months.png)','![지급 구현별 차이](figures/payoff_implementation_difference.png)','',
      '## MC 수치 안정성','',table(conv.rename(columns={'n':'변경 시나리오수','mean_abs_change':'2만→4만 평균 절대변화(원)','p95_abs_change':'절대변화 95백분위(원)','mean_mc_se':'4만×3시드 증분 SE 평균(원)'})),
      '표의 MC 오차와 2만→4만 경로 변화는 가격 계산의 수치 안정성을 뜻한다. DeepONet 예측 오차와는 다른 지표다. 각 시드별 평균 증분은 `results/mc_effect_by_seed.csv`에 저장했다.','']
    if model is not None:
        summary,selected,nfail=model;show=summary[summary.cohort.isin(['regular_test','monthly_test'])].copy()
        show['price_r2']=show.price_r2.map(lambda x:f'{x:.4f}')
        for col in ['resolved_sign','within_1','within_5','within_10']:show[col]*=100
        display=show.rename(columns={'payoff':'구현','cohort':'시험군','scope':'범위','model':'학습 방식','price_r2':'가격 R²','price_mae':'가격 MAE(원)','delta_mae':'증분 MAE(원)','resolved_sign':'방향 일치율(%)','within_1':'1원 이내(%)','within_5':'5원 이내(%)','within_10':'10원 이내(%)'})
        lines+=['## DeepONet 학습·평가','',
          '각 지급 구현의 MC 라벨로 가격 손실 / 가격+증분 손실 / 가격+증분 손실+쿠폰 선형 구조를 각각 학습했다. 방식당 3시드, 총 18개 모델이다. 모든 모델은 기존 고정 하이퍼파라미터를 사용했고 검증 점수로 선택한 뒤 시험했다. PI 손실과 Stage 2는 사용하지 않았다.','',
          '선택 방식: '+', '.join(f'`{a}` → `{b}`' for a,b in selected.items())+'.','',
          '가격 지표는 기준·변경 가격 전체, 증분 지표는 변경 시나리오로 계산했다. `terms`는 쿠폰·KI·행사가·월 쿠폰 배리어, `schedule`은 관측 추가·만기 변경이다. 아래 결과는 각 방식의 3개 학습 시드 평균 예측이다. 별도 기준 계약 가격 지표는 `model_baseline_metrics.csv`에 있다.','',
          table(display[['구현','시험군','범위','학습 방식','가격 R²','가격 MAE(원)','증분 MAE(원)','방향 일치율(%)','1원 이내(%)','5원 이내(%)','10원 이내(%)']]),
          '![가격 예측](figures/model_prices.png)','![계약조건 증분 예측](figures/model_increment_terms.png)','![일정 증분 예측](figures/model_increment_schedule.png)','',
          f'선택 모델의 MC로 구분 가능한 방향 실패는 시험·진단을 합쳐 **{nfail:,}개 시나리오**다. 분모와 상품·축별 지표는 `model_metrics.csv`, 개별 실패는 `resolved_direction_failures.csv.gz`에 있다. 이 수만으로 모든 조건에서 안정적이라고 판정하지 않는다.','',
          '방향 평가는 |ΔMC| > 1.96×paired MC SE이면서 영향 경로가 30개 이상인 경우만 사용했다. 1/5/10원 이내 비율은 오차 분포를 보기 위한 진단값이다. 시드·상품·변경 폭별 오차와 일정 변경 실패를 함께 확인해야 한다.','',
          '기존 동결 모델과의 차이에는 IV·곡선·학습 라벨 변경과 재학습 효과가 함께 들어간다. 같은 라벨 내 학습 방식 비교만 손실·구조 변경에 대한 통제 비교이며, 새 자료 효과를 독립적으로 식별한 실험은 아니다. 기존 시험 계약을 다시 사용하므로 완전히 새로운 독립 시험이라고 부르지 않는다.','']
        lines += ['### 평균 오차의 개선과 남은 실패','']
        for payoff,arm in selected.items():
            for cohort in ['regular_test','monthly_test']:
                g=summary[summary.payoff.eq(payoff)&summary.cohort.eq(cohort)]
                old=g[g.model.eq('augmented_price')&g.scope.eq('terms')].iloc[0]
                new=g[g.model.eq(arm)&g.scope.eq('terms')].iloc[0]
                sch=g[g.model.eq(arm)&g.scope.eq('schedule')].iloc[0]
                lines.append(f'- `{payoff}` / {cohort}: 가격만 학습한 모델 → 선택 모델의 계약조건 증분 MAE **{old.delta_mae:.2f} → {new.delta_mae:.2f}원**. 선택 모델의 가격 R² **{new.price_r2:.4f}**, 방향 일치율 **{100*new.resolved_sign:.2f}%**. 관측·만기 변경의 증분 MAE는 **{sch.delta_mae:.2f}원**이다.')
        lines += ['', '검증 점수로 선택한 방식이 모든 시험 축의 최소 오차를 보장하지 않는다. 이 실험에서 계약조건 증분의 평균 오차는 감소했으나 방향 불일치와 일정 변경 오차가 남았으므로, **모든 상품·변경 폭에서 가격 증분을 안정적으로 예측한다고 결론 내릴 수 없다.** 지급 구현을 상세화하는 것과 그 라벨을 DeepONet이 잘 근사하는 것은 각각 확인해야 한다.','',
          '선택 방식의 계약조건·변경 폭별 결과는 `model_selected_axis.csv`와 각 구현의 `metrics_by_offset.csv`, 학습 시드별 결과는 `model_seed_summary.csv`, 구조별 결과는 `model_metrics_by_structure.csv`에 있다. 아래 95% bootstrap 구간은 계약별 평균 증분 오차 차이를 구하고 각 계약에 같은 가중치를 주어 계산했다. 따라서 변경 시나리오에 같은 가중치를 주는 위 전체 MAE 차이와는 조금 다를 수 있다. 음수는 선택 방식의 오차가 더 작다는 뜻이다.','',table(pd.read_csv(OUT/'paired_model_improvement.csv'))]
    else:lines+=['## DeepONet 학습·평가','', '**아직 완료되지 않음.** MC 라벨 완료 이후 두 지급 구현 각각 학습·평가한다.','']
    lines+=['## 재현 파일','',
      '`prepare.py` → `run_population.py` → `plot.py --fresh`; `synthetic.py prepare` → `synthetic.py verify` → `synthetic.py run` → `learn.py train` → `learn.py evaluate` → `report.py`. 원본 입력·코드 해시는 `source_manifest.json`, MC 설정은 `protocol.json`, 실행 로그·완료 표시는 `results/`에 있다. 저장 결과가 있으면 plot/report만 실행하며 MC를 반복하지 않는다.','',
      '원본 폴더는 읽어서 복사만 했으며 계산과 결과 저장은 이 branch에서 수행했다. 이 비교는 정의된 GBM·지급 구현의 MC 반사실 실험이다. 모든 실제 상품 약관의 정확성이나 관측 공정가의 인과효과를 검증한 결과는 아니다.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines));print('REPORT generated; models complete',model is not None,'population complete',(OUT/'population_complete.json').exists(),flush=True)

if __name__=='__main__':main()
