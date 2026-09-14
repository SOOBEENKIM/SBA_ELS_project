"""Generate the IV report solely from recorded experiment outputs."""
from pathlib import Path
import json,sys,gzip,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import HERE,ROOT,V3,OUT,dump,preserved
from run import metrics

AXES={'coupon_regular':'연 쿠폰','ki_barrier':'낙인 배리어','first_strike':'1차 행사가','last_strike':'만기 행사가',
      'monthly_barrier':'월 쿠폰 지급 배리어','nobs_early':'첫 평가 전 관측 추가','nobs_middle':'중간 관측 추가',
      'nobs_late':'마지막 구간 관측 추가','tenor_months':'만기'}
COHORT={'regular_test':'일반형 시험','monthly_test':'월지급형 시험','published_monthly':'공시 월지급형'}
def table(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def num(x,d=2):return '—' if pd.isna(x) else f'{x:,.{d}f}'
def pct(x):return '—' if pd.isna(x) else f'{100*x:.1f}%'

def main():
    cfg=json.loads((HERE/'protocol.json').read_text());audit=json.loads((OUT/'input_audit.json').read_text())
    complete=json.loads((OUT/'complete.json').read_text());assert complete['status']=='complete'
    assert preserved()==cfg['preserved_inputs']
    fs=json.loads((OUT/'families.json').read_text());labels=pd.read_csv(OUT/'mc_labels.csv')
    pred=pd.read_csv(OUT/'predictions.csv.gz',dtype={'model_seed':str});ms=pd.read_csv(OUT/'model_metrics.csv',dtype={'model_seed':str})
    assignment=pd.read_csv(OUT/'market_assignment.csv');primary=cfg['frozen_models']['primary']
    primary_pred=pred[pred.model.eq(primary)&pred.model_seed.eq('ensemble')].copy()
    primary_pred.to_csv(OUT/'primary_predictions.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    labels['cohort']=np.where(labels.origin.eq('published_monthly'),'published_monthly',np.where(labels.monthly,'monthly_test','regular_test'))
    labels['halfwidth_95_krw']=1.96*labels.delta_se_krw
    labels['resolved']=(abs(labels.delta_krw)>labels.halfwidth_95_krw)&(labels.affected>=30)
    effects=[]
    for key,g in labels[labels.axis.ne('base')].groupby(['cohort','arm','axis','offset']):
        effects.append(dict(zip(['cohort','arm','axis','offset'],key),n=len(g),mean_delta_krw=g.delta_krw.mean(),
            median_delta_krw=g.delta_krw.median(),min_delta_krw=g.delta_krw.min(),max_delta_krw=g.delta_krw.max(),
            median_mc_halfwidth=g.halfwidth_95_krw.median(),max_mc_halfwidth=g.halfwidth_95_krw.max(),
            resolved_fraction=g.resolved.mean(),positive=int((g.delta_krw>1e-8).sum()),negative=int((g.delta_krw< -1e-8).sum())))
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'mc_effect_summary.csv',index=False)
    pooled=pd.read_csv(OUT/'mc_checkpoints.csv.gz');raw=pd.read_csv(OUT/'mc_seed_checkpoints.csv.gz')
    conv=pooled[pooled.paths_per_seed.eq(20000)].merge(pooled[pooled.paths_per_seed.eq(40000)],on=['family','arm','case'],suffixes=('_20k','_40k'))
    conv['delta_change_20k_40k']=conv.delta_krw_40k-conv.delta_krw_20k
    conv.to_csv(OUT/'mc_convergence.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    rf=raw[raw.paths.eq(40000)].copy();rf['seed_delta_krw']=rf.delta_sum/rf.paths*10000
    dispersion=rf.groupby(['family','arm','case']).seed_delta_krw.agg(['min','max','std']).reset_index()
    dispersion['seed_range_krw']=dispersion['max']-dispersion['min'];dispersion.to_csv(OUT/'mc_seed_dispersion.csv',index=False)
    mdisp=pred[pred.model_seed.ne('ensemble')].groupby(['family','arm','case','model']).pred_delta_krw.agg(['min','max','std']).reset_index()
    mdisp['seed_range_krw']=mdisp['max']-mdisp['min'];mdisp.to_csv(OUT/'model_seed_dispersion.csv',index=False)

    # Checks address the changed experiment boundary rather than re-testing all legacy code.
    assert len(labels)==audit['cases_per_arm']*3
    assert len(pred)==audit['cases_per_arm']*2*12
    np.testing.assert_allclose(labels.price_krw,labels.monthly_pv_krw+labels.nonmonthly_pv_krw,atol=1e-8,rtol=0)
    assert raw.groupby('family').seed.nunique().eq(3).all()
    for r in fs:
        if r['origin']=='published_monthly':
            for k in ['sigs','corr','beta','dividend_yields']:
                np.testing.assert_allclose(r['markets']['HV'][k],r['original_market'].get(k,[0.,0.,0.]),atol=1e-12,rtol=0)
    for _,g in labels[labels.arm.isin(['HV','IV'])&labels.axis.eq('coupon_regular')].groupby(['family','arm']):
        slopes=g.delta_krw/g.offset;assert slopes.min()>=-1e-7
        np.testing.assert_allclose(slopes,slopes.iloc[0],atol=1e-6,rtol=1e-9)
    for axis in ['ki_barrier','monthly_barrier']:
        g=labels[labels.arm.isin(['HV','IV'])&labels.axis.eq(axis)]
        assert (g.delta_krw*g.offset<=1e-7).all(),axis
    aff=pred[pred.model.eq('affine_coupon_delta')&pred.axis.eq('coupon_regular')]
    assert (aff.pred_delta_krw*aff.offset>=-1e-6).all()
    assert raw.groupby(['family','case','replicate','paths']).seed.nunique().eq(1).all()
    assert labels.loc[labels.axis.eq('base'),'delta_krw'].eq(0).all()
    rawcov=pd.read_csv(OUT/'universe_iv_coverage.csv').set_index('item')
    exclusions=pd.read_csv(OUT/'excluded_families.csv');exclusions['detail']=exclusions.item.map(rawcov.reason)
    exclusions.to_csv(OUT/'excluded_families_detailed.csv',index=False)
    dump(OUT/'post_run_verification.json',dict(status='pass',existing_artifacts_unchanged=True,
        all_expected_scenarios=True,three_independent_seeds=True,paired_gaussian_seeds=True,
        coupon_affine_nonnegative=True,ki_and_monthly_barrier_pathwise_direction=True,
        frozen_affine_coupon_direction=True,base_delta_exactly_zero=True))
    boot=[];rng=np.random.default_rng(2026091402)
    for cohort in ['regular_test','monthly_test','published_monthly']:
        for suite in ['terms','schedule']:
            g=primary_pred[primary_pred.cohort.eq(cohort)&primary_pred.suite.eq(suite)&primary_pred.axis.ne('base')]
            means=g.assign(ae=abs(g.delta_error_krw)).groupby(['family','arm']).ae.mean().unstack()
            d=(means.IV-means.HV).to_numpy();draws=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
            boot.append(dict(cohort=cohort,suite=suite,n=len(d),iv_minus_hv_family_delta_mae=float(d.mean()),
                ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975))))
    pd.DataFrame(boot).to_csv(OUT/'paired_model_error_bootstrap.csv',index=False)

    figs=HERE/'figures';figs.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'figure.dpi':140})
    choice=[('coupon_regular',.01,'Coupon +1pp'),('ki_barrier',.05,'KI +5pp'),('first_strike',.05,'First strike +5pp'),('last_strike',.05,'Last strike +5pp'),('monthly_barrier',.05,'Monthly barrier +5pp')]
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for ax,cohort in zip(axs,['regular_test','monthly_test']):
        opts=choice[:4] if cohort=='regular_test' else choice;x=np.arange(len(opts))
        for arm,shift,color in [('HV',-.18,'#557fa5'),('IV',.18,'#de8739')]:
            vals=[effects.loc[effects.cohort.eq(cohort)&effects.arm.eq(arm)&effects.axis.eq(a)&np.isclose(effects.offset,h),'mean_delta_krw'].iloc[0] for a,h,_ in opts]
            ax.bar(x+shift,vals,.35,label=arm,color=color)
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,[z[2] for z in opts],rotation=22,ha='right');ax.set_ylabel('Mean MC increment (KRW / 10,000)');ax.set_title(cohort);ax.legend()
    fig.tight_layout();fig.savefig(figs/'mc_increments.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,suite in zip(axs,['terms','schedule']):
        for arm,shift,color in [('HV',-.18,'#557fa5'),('IV',.18,'#de8739')]:
            vals=[ms.loc[ms.cohort.eq(c)&ms.arm.eq(arm)&ms.model.eq(primary)&ms.model_seed.eq('ensemble')&ms.axis.eq(suite),'delta_mae'].iloc[0] for c in ['regular_test','monthly_test']]
            ax.bar(np.arange(2)+shift,vals,.35,label=arm,color=color)
        ax.set_xticks([0,1],['Regular test','Monthly test']);ax.set_title(suite+' increments');ax.set_ylabel('Frozen DeepONet increment MAE (KRW)');ax.legend()
    fig.tight_layout();fig.savefig(figs/'frozen_deeponet_errors.png');plt.close(fig)

    rows=[]
    for cohort in ['regular_test','monthly_test','published_monthly']:
        for arm in ['HV','IV']:
            g=assignment[(assignment.origin.eq('published_monthly') if cohort=='published_monthly' else assignment.origin.eq('synthetic_monthly' if cohort=='monthly_test' else 'synthetic_regular'))&assignment.arm.eq(arm)]
            rows.append([COHORT[cohort],arm,len(g),int(g.sigma_outside_training.sum()),num(g.min_sigma.min()*100),num(g.max_sigma.max()*100)])
    text=['# ATM IV 적용 MC 가격·증분 및 고정 DeepONet 검증',
      '\n예측 타깃은 MC 이론가다. 관측 공정가 학습, Stage 2 잔차 모델, PI 손실, 신규 모델 학습은 사용하지 않았다. 기존 계약 구현과 9개 DeepONet 가중치를 보존하고 시장 입력을 변경한 평가다.',
      '\n## 데이터와 비교 설계',
      f"첨부 ZIP의 {audit['csv_files']:,}개 CSV, {audit['valid_rows']:,}개 행, {audit['notion_series']}개 notion RIC를 확인했다. 기간은 {audit['first_date']}~{audit['last_date']}다. 파일은 `calc_date/notion_ric/source_ric/iv`만 포함하며 옵션 만기 정보는 없다. IV는 소수 연율로 해석하고 기초자산별 고정 ATM 변동성으로 사용했다. 기간구조나 스마일을 복원한 실험은 아니다.",
      f"기존 262개 시험·공시 계약 중 {audit['included_families']}개를 포함하고 {audit['excluded_families']}개 월지급형 시험 계약을 제외했다. 세 자산의 발행일 이전 또는 당일 IV가 모두 있어야 하며 최대 7달력일 이내 값만 허용했다. 제외는 IV 이력 시작 전 발행 또는 미확보 자산 때문이며 상세 사유를 CSV에 기록했다. IV 미확보 부분을 역사적 변동성으로 대체하지 않았다.",
      '일반형 시험 계약 128개에는 적격 실상품에서 사전 고정 난수로 뽑은 자산·발행일의 시장 상태를 부여했다. 월지급형 시험 계약은 원래 일정 템플릿의 자산·발행일을 사용했다. 두 경우 모두 기존 합성 계약의 지급 규칙을 유지한다. 시장 상태를 제공한 실상품의 공정가를 합성 계약의 실제 가격으로 사용하지 않았다. 공시 6개는 원래 자산·발행일과 검증된 계약을 유지했다.',
      'HV와 IV 두 군은 변동성만 다르다. HV는 발행 전 최대 180거래일 로그수익률의 연율 표준편차이고, 두 군의 상관계수는 동일한 역사적 상관이다. 금리·NS 방식·q=0·계약·난수·지급 일정은 동일하다. 날짜 단위 as-of이며 개별 데이터의 장중 가용 시각 또는 FRED 공표 빈티지를 검증한 것은 아니다.',
      '**이 HV군은 이전 합성 시험의 임의 시장 상태를 그대로 재사용한 군이 아니다. 이번에 구성한 실제 시장 상태의 HV/IV 쌍이다. 이전 Report_2 평균과 이번 IV 평균의 차이를 IV 단독 효과로 해석하지 않는다.**',
      table(['평가군','변동성','계약 수','학습 변동성 범위 밖 계약','최소 σ(%)','최대 σ(%)'],rows),
      '\n범위 밖은 적어도 한 자산의 σ가 기존 합성 학습 범위 12~45% 밖이라는 뜻이다. 다른 입력의 분포 차이까지 판정한 지표는 아니다. 원본에는 IV>300%인 행도 있으나 이번 선정 시장의 최대 IV는 위 표와 같으며, 수치에 맞추는 clipping은 하지 않았다.',
      '\n## MC 계산과 계약 변경',
      f"각 군 {audit['cases_per_arm']:,}개 기준·변경 시나리오, 시드당 40,000경로×3시드다. 10,000/20,000/40,000경로는 누적 체크포인트다. HV와 IV, 기준과 변경 계약에 같은 표준정규 난수를 사용했다. 각 가격과 증분뿐 아니라 IV−HV 가격·증분 차이의 paired 표준오차도 저장했다.",
      '쿠폰 ±0.1/0.25/0.5/1%p, KI·1차/만기 행사가·월 지급 배리어 ±0.5/1/2.5/5%p와 기존 일정 실험의 관측 추가 3위치·만기 ±1/3/6/12개월을 포함했다. 관측 추가와 만기 변경의 일정·이자기간 규칙은 기존 V4 계약을 그대로 읽었다. 월 쿠폰 지급 배리어와 KI는 별개다. 만기를 바꿔도 이번에는 단일 IV 값을 고정한다.',
      '\n## MC 조건별 가격 증분',
      '표의 값은 같은 군 안에서 `변경 계약 MC − 기준 계약 MC`를 먼저 구한 뒤 계약별로 평균한 원 단위 값이다. 액면 10,000원 기준이다.']
    erows=[]
    for cohort in ['regular_test','monthly_test']:
        for a,h,_ in choice+[('nobs_early',1,''),('nobs_middle',1,''),('nobs_late',1,''),('tenor_months',6,'')]:
            g=effects[effects.cohort.eq(cohort)&effects.axis.eq(a)&np.isclose(effects.offset,h)]
            if not len(g):continue
            z=g.set_index('arm');label='+1%p' if a=='coupon_regular' else '+6개월' if a=='tenor_months' else '+1회' if a.startswith('nobs_') else '+5%p'
            erows.append([COHORT[cohort],AXES[a]+' '+label,int(z.at['HV','n']),num(z.at['HV','mean_delta_krw']),num(z.at['IV','mean_delta_krw']),num(z.at['IV_minus_HV','mean_delta_krw'])])
    text += [table(['평가군','변경','계약 수','HV 증분','IV 증분','IV−HV 증분'],erows),
        '\n![MC increments](figures/mc_increments.png)',
        '\n전체 변경 폭, 계약별 부호, 범위 및 MC 불확실성은 `results/mc_effect_summary.csv`와 `results/mc_labels.csv`에 있다. 평균 증분은 모든 계약의 동일 부호를 뜻하지 않는다. 1차 행사가·일정 변경의 방향을 임의로 강제하지 않았다.',
        '\n## 고정 DeepONet 평가',
        '기존 3개 학습 방식×3개 시드를 모두 평가했다. 대표 결과는 이전 검증군에서 이미 선택된 `affine_coupon_delta` 3시드 평균이며 이번 IV 결과로 모델을 다시 선택하지 않았다. 두 군 모두 해당 군의 변동성을 모델 입력과 MC 입력에 동일하게 적용했다.']
    mr=[]
    for cohort in ['regular_test','monthly_test','published_monthly']:
        for suite in ['terms','schedule']:
            for arm in ['HV','IV']:
                g=ms[ms.cohort.eq(cohort)&ms.axis.eq(suite)&ms.arm.eq(arm)&ms.model.eq(primary)&ms.model_seed.eq('ensemble')].iloc[0]
                mr.append([COHORT[cohort],'계약조건' if suite=='terms' else '관측·만기',arm,num(g.delta_mae),num(g.delta_rmse),pct(g.resolved_sign),pct(g.within_5)])
    text += [table(['평가군','증분 종류','입력','증분 MAE(원)','RMSE(원)','구분 가능한 부호 일치','5원 이내'],mr),
       '\n부호 일치율은 |MC 증분|>1.96×paired SE이고 영향받은 경로가 30개 이상인 경우만 계산한다. 군별로 이 분모가 달라질 수 있으므로 MAE와 함께 본다. 가격 수준의 R²만으로 증분 안정성을 판단하지 않는다.',
       '\n![Frozen DeepONet errors](figures/frozen_deeponet_errors.png)',
       '\n## 결과 해석']
    for cohort in ['regular_test','monthly_test']:
        for suite in ['terms','schedule']:
            g=ms[ms.cohort.eq(cohort)&ms.axis.eq(suite)&ms.model.eq(primary)&ms.model_seed.eq('ensemble')].set_index('arm')
            hv=g.at['HV','delta_mae'];ivv=g.at['IV','delta_mae'];word='감소' if ivv<hv else '증가'
            text.append(f"- {COHORT[cohort]}의 {'쿠폰·배리어·행사가' if suite=='terms' else '관측·만기'} 증분 MAE는 HV {hv:.2f}원 → IV {ivv:.2f}원으로 {word}했다. 이 값은 같은 계약 표본의 고정 모델 비교다.")
    bb=[]
    for cohort in ['regular_test','monthly_test','published_monthly']:
        for arm in ['HV','IV']:
            g=ms[ms.cohort.eq(cohort)&ms.axis.eq('base')&ms.model.eq(primary)&ms.model_seed.eq('ensemble')&ms.arm.eq(arm)].iloc[0]
            bb.append([COHORT[cohort],arm,num(g.price_mae),num(g.price_rmse),num(g.price_r2,4)])
    text += ['\n기준 계약의 가격 수준 성능:',table(['평가군','입력','가격 MAE(원)','RMSE(원)','R²'],bb),
      '\nIV를 적용했을 때 MC 가격이 달라진 것과 DeepONet의 근사 오차가 개선된 것은 별개의 결과다. 본 실험은 ATM IV 입력에서도 기존 모델이 기준 MC를 근사하는지 평가했으며, 실무 견적 가격의 정확성이나 모든 계약에서의 안정성을 입증한 것은 아니다. 변동성 범위 밖 계약 및 모델별·시드별 차이는 별도 CSV로 보존했다.',
      '\n## MC 자체의 수치 불확실성']
    sr=[]
    for cohort in ['regular_test','monthly_test']:
        for arm in ['HV','IV']:
            g=labels[labels.cohort.eq(cohort)&labels.arm.eq(arm)&labels.axis.ne('base')]
            sr.append([COHORT[cohort],arm,num(g.halfwidth_95_krw.median()),num(g.halfwidth_95_krw.max()),pct(g.resolved.mean())])
    text += [table(['평가군','입력','95% 반폭 중앙값(원)','최대 반폭(원)','부호 구분 가능 비율'],sr),
      '\n95% 반폭은 각 증분 MC 추정치의 정규근사 수치 불확실성이다. 상품 모집단의 신뢰구간이나 모델 위험의 범위가 아니다. 계약별 경계 사건에서는 불확실성이 커질 수 있으므로 단일 방향 불일치만으로 구현 오류를 단정하지 않는다.',
      '\n## 검증과 재현',
      '- 신규 paired 계산기를 기존 V4 엔진과 일반형 1개·월지급형 1개의 모든 변경 시나리오에서 각 500경로로 비교했고 합계 오차는 0이었다. HV=IV이면 가격·증분 차이와 그 분산이 정확히 0이었다.',
      '- 공시 6개의 새 HV 시장 입력을 기존 공시 실험의 시장 입력과 대조했고 1e-12 이내로 일치했다.',
      '- 계약 변경·시장 고정·IV as-of·세 시드·현금흐름 분해·가격 차이와 증분의 일치 및 쿠폰 선형성을 확인했다.',
      '- 기존 MC 엔진, 입력 인코더, 학습 자료와 9개 모델 가중치의 해시가 변경되지 않았음을 확인했다.',
      '- `protocol.json`, `results/input_audit.json`, `results/pre_mc_verification.json`, `results/post_run_verification.json`에 설정과 검증을 저장했다. 실행 방법은 이 폴더의 README를 참조한다.',
      '\nMC 이론가가 최종 예측 타깃인 방향을 유지한다. 이후 학습을 추가한다면 현재 평가와 분리된 학습·검증 계약 및 시장 상태를 구성하고, 독립 시험에서 가격·증분 오차를 다시 확인해야 한다.']
    alignment=OUT/'prediction_alignment_verification.json'
    if alignment.exists():
        a=json.loads(alignment.read_text());assert a['status']=='pass'
        text.insert(-1,f"- 모델별로 {a['independently_encoded_prediction_pairs']}개 가격·증분 쌍을 따로 인코딩해 저장 결과와 대조했다. 작은 배치와 원래 큰 배치의 최대 차이는 {a['max_delta_difference_krw']:.6f}원이었다. 입력 연결 오류와 모델의 예측 오차를 구분하기 위한 확인이다.")
    (HERE/'REPORT.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    # Lossless compact audit payload for version control; keep the working JSON.
    with (OUT/'families.json').open('rb') as src,gzip.GzipFile(filename=str(OUT/'families.json.gz'),mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
    print('REPORT',HERE/'REPORT.md',flush=True)
    print(ms[ms.model.eq(primary)&ms.model_seed.eq('ensemble')&ms.axis.isin(['terms','schedule'])].to_string(index=False),flush=True)

if __name__=='__main__':main()
