"""Render tables, commentary and the figure from recorded experiment results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = Path('analysis/mc_monthly_v3_20260910')
V4 = Path('analysis/mc_schedule_v4_20260911')
ARM_NAMES = {
    'augmented_price': '합성 가격 손실',
    'augmented_delta': '가격 + 증분 손실',
    'affine_coupon_delta': '가격 + 증분 손실 + 쿠폰 선형 구조',
}


def table(cols, rows):
    return '\n'.join(['| ' + ' | '.join(cols) + ' |',
                      '|' + '|'.join(['---'] * len(cols)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def select(frame, **filters):
    mask = np.ones(len(frame), dtype=bool)
    for key, value in filters.items():
        mask &= np.isclose(frame[key], value) if isinstance(value, float) else frame[key].eq(value)
    return frame.loc[mask]


def one(frame, **filters):
    rows = select(frame, **filters)
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one result for {filters}, found {len(rows)}')
    row = rows.iloc[0]
    for key, value in row.items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError(f'Non-finite reported result: {filters}, {key}={value}')
    return row


def unique_int(values, label):
    values = pd.Series(values).drop_duplicates()
    if len(values) != 1 or not np.isfinite(values.iloc[0]) or int(values.iloc[0]) != values.iloc[0]:
        raise ValueError(f'Expected one integer {label}, found {values.tolist()}')
    return int(values.iloc[0])


def change_word(before, after):
    if abs(after - before) < 1e-10:
        return '동일했다'
    return '감소했다' if after < before else '증가했다'


def signed(value):
    return f'{value:+.2f}'.replace('-', '−')


def render(root=ROOT):
    """Read and validate referenced results before writing any report."""
    root = Path(root)
    a, b = root / V3 / 'results', root / V4 / 'results'
    cfg = json.loads((root / V3 / 'protocol.json').read_text())
    schedule_cfg = json.loads((root / V4 / 'protocol.json').read_text())
    selection = json.loads((a / 'model_selection.json').read_text())
    primary = selection['selected_arm']
    arms = cfg['learning']['arms']
    if primary not in arms or any(arm not in ARM_NAMES for arm in arms):
        raise ValueError(f'Unknown selected/model arm: {primary}, {arms}')
    labels = pd.read_csv(a / 'mc_labels.csv')
    metrics = pd.read_csv(a / 'model_metrics.csv')
    sm = pd.read_csv(b / 'model_metrics.csv')
    effects = pd.read_csv(b / 'mc_effect_summary.csv')
    schedule_labels = pd.read_csv(b / 'mc_labels.csv')
    predictions = pd.read_csv(a / 'predictions.csv')
    sp = pd.read_csv(b / 'predictions.csv')
    independent = pd.read_csv(root / 'analysis/mc_v3_audit_20260910/results/independent_mc.csv')
    base = select(labels, axis='base')
    counts = base.groupby('split').family.nunique()
    public_count = select(base, origin='published_monthly').family.nunique()
    reg_n = select(base, split='test', origin='synthetic_regular').family.nunique()
    mon_n = select(base, split='test', origin='synthetic_monthly').family.nunique()
    paths = unique_int(labels.paths_per_seed, 'paths per seed')
    train_reps = unique_int(labels.loc[labels.split.isin(['train', 'validation']), 'replicates'], 'train/validation replicates')
    test_reps = unique_int(labels.loc[labels.split.isin(['test', 'stress']), 'replicates'], 'test/stress replicates')
    schedule_paths = unique_int(schedule_labels.paths_per_seed, 'schedule paths per seed')
    schedule_reps = unique_int(schedule_labels.replicates, 'schedule replicates')
    seeds = cfg['learning']['seeds']
    checkpoint_count = len(selection['checkpoint_hashes'])
    if checkpoint_count != len(arms) * len(seeds):
        raise ValueError('Model count and recorded checkpoint count disagree')

    rows = []
    for axis, h, title in [('coupon_regular', .01, '연 쿠폰 +1%p'), ('ki_barrier', .05, '낙인 배리어 +5%p'),
                           ('first_strike', .05, '1차 행사가 +5%p'), ('last_strike', .05, '만기 행사가 +5%p'),
                           ('monthly_barrier', .05, '월 쿠폰 지급 배리어 +5%p')]:
        row = [title]
        for origin in ['synthetic_regular', 'synthetic_monthly']:
            q = select(labels, origin=origin, split='test', axis=axis, offset=h)
            if q.empty and not (axis == 'monthly_barrier' and origin == 'synthetic_regular'):
                raise ValueError(f'Missing counterfactual result: {origin}, {axis}, {h}')
            row.append('해당 없음' if q.empty else f'{q.delta_krw.mean():+.2f}원 ({q.family.nunique()}개)')
        rows.append(row)

    selected, modelrows, improvement = {}, [], []
    for cohort, title in [('regular_test', '일반형'), ('monthly_test', '월지급형')]:
        for arm in arms:
            r = one(metrics, cohort=cohort, model=arm, seed='ensemble', axis='ALL')
            name = ARM_NAMES[arm] + (' (선택)' if arm == primary else '')
            modelrows.append([title, name, f'{r.price_r2:.4f}', f'{r.price_mae:.2f}', f'{r.delta_mae:.2f}', f'{r.resolved_sign*100:.2f}%'])
        selected[cohort] = one(metrics, cohort=cohort, model=primary, seed='ensemble', axis='ALL')
        comparator = one(metrics, cohort=cohort, model='augmented_price', seed='ensemble', axis='ALL')
        before, after = comparator.delta_mae, selected[cohort].delta_mae
        improvement.append(f'{title} **{before:.2f}→{after:.2f}원**으로 {change_word(before, after)}.')
    rank = select(metrics, cohort='monthly_test', seed='ensemble', axis='ALL')
    best = rank[rank.model.isin(arms)].sort_values(['delta_mae', 'model']).iloc[0]
    ranking_text = (f'월지급형 증분 MAE가 가장 작은 방식은 {ARM_NAMES[best.model]}({best.delta_mae:.2f}원)이며, '
                    f'검증 기준으로 선택된 방식은 {ARM_NAMES[primary]}({selected["monthly_test"].delta_mae:.2f}원)이다.')

    sr = []
    schedule_axes = [('nobs_early', 1, '관측 +1회: 첫 평가 전'), ('nobs_middle', 1, '관측 +1회: 중간'),
                     ('nobs_late', 1, '관측 +1회: 마지막 구간')]
    schedule_axes += [('tenor_months', h, f'만기 {h:+d}개월') for h in schedule_cfg['tenor_month_offsets']]
    for axis, h, title in schedule_axes:
        rr = [title]
        for cohort in ['regular_test_128', 'monthly_test_128']:
            x = one(effects, cohort=cohort, axis=axis, offset=h)
            y = one(sm, cohort=cohort, model=primary, training_seed='ensemble', axis=axis, offset=h)
            rr.extend([f'{x["mean"]:+.2f}', f'{y.delta_mae:.2f}'])
        sr.append(rr)
    reg_obs = int(one(effects, cohort='regular_test_128', axis='nobs_early', offset=1).n_families)
    mon_obs = int(one(effects, cohort='monthly_test_128', axis='nobs_early', offset=1).n_families)

    case_id = 'published_KR6MD0003TN4'
    case = one(predictions, family=case_id, model=primary, seed='ensemble', axis='first_strike', offset=.05)
    check = one(independent, family=case_id)
    direction = '같았다' if np.sign(case.mc_delta_krw) == np.sign(case.pred_delta_krw) else '달랐다'
    schedule_notes = []
    for axis, title in [('nobs_middle', '중간'), ('nobs_late', '마지막')]:
        x = one(effects, cohort='monthly_test_128', axis=axis, offset=1)
        y = one(sm, cohort='monthly_test_128', model=primary, training_seed='ensemble', axis=axis, offset=1)
        q = select(sp, cohort='monthly_test_128', model=primary, training_seed='ensemble', axis=axis, offset=1)
        if q.family.nunique() != int(x.n_families) or len(q) != int(x.n_families):
            raise ValueError(f'Incomplete or duplicate schedule predictions for {axis}')
        schedule_notes.append(f'{title} 구간: MC 평균 **{signed(x["mean"])}원**, 모델 평균 **{signed(q.pred_delta_krw.mean())}원**, 증분 MAE **{y.delta_mae:.2f}원**')

    selected_predictions = select(predictions, model=primary, seed='ensemble', split='test')
    selected_schedule = select(sp, model=primary, training_seed='ensemble')
    selected_schedule = selected_schedule[selected_schedule.cohort.isin(['regular_test_128', 'monthly_test_128'])]
    disagreements = 0
    for p in [selected_predictions, selected_schedule]:
        p = p[p.axis.ne('base')]
        resolved = p.mc_delta_krw.abs().gt(1.96*p.mc_delta_se_krw) & p.affected.ge(30)
        disagreements += int((resolved & (np.sign(p.mc_delta_krw) != np.sign(p.pred_delta_krw))).sum())
    verdict = (f'MC 오차로 방향을 구분할 수 있는 시험 변경 시나리오 중 **{disagreements:,}건**에서 선택 모델과 MC의 증분 방향이 달랐다.'
               if disagreements else 'MC 오차로 방향을 구분할 수 있는 시험 변경 시나리오에서 선택 모델과 MC의 증분 방향 불일치는 관측되지 않았다.')
    checkpoints = '→'.join(f'{n:,}' for n in cfg['mc']['checkpoints'])
    text = f'''## 최종 실험 결과

아래 결과는 수정된 계약 현금흐름을 사용한 실제 실행 결과다. 과거 잘못된 MC 구현의 수치는 제외했다. **모든 원화 금액은 액면 {cfg['notional']:,}원 기준**이며, 실제 관측 공정가의 인과효과가 아니라 **정의된 MC 모형 안에서 계약조건을 변경한 반사실 가격 차이**다. 쿠폰·배리어·행사가 실험은 계약별 시장·일정·나머지 조건을 고정하고, 관측 횟수·만기 실험은 명시된 규칙에 따라 종속 일정과 쿠폰 이자기간도 조정했다.

### 완료된 실행 범위

- 계약조건 실험: 기준 계약 {base.family.nunique():,}개, 기준·변경 시나리오 {len(labels):,}개. 학습 {counts['train']:,}개 / 검증 {counts['validation']:,}개 / 시험 {counts['test']:,}개 / 진단 {counts['stress']:,}개(공시 확인 월지급형 {public_count}개 + 이전 참고 계약 {counts['stress']-public_count}개).
- 일반형 시험 {reg_n}개·월지급형 시험 {mon_n}개. 월지급형은 계약 템플릿 그룹을 분리하고 공시 비교군의 템플릿도 학습에서 제외했다.
- MC는 **시드당 {paths:,}경로**. 학습·검증은 {train_reps}시드, 시험·진단은 {test_reps}개 독립 시드(계약당 합계 {paths*test_reps:,}경로). {checkpoints}은 같은 실행의 누적 체크포인트다.
- {len(arms)}개 학습 방식 × 학습 시드 {'·'.join(map(str,seeds))} = {checkpoint_count}개 DeepONet. 모델 선택은 검증 자료로 끝냈고 선택 방식의 {len(seeds)}시드 평균을 사용했다. PI 손실과 Stage 2는 사용하지 않았다.
- 관측 횟수·만기 추가 평가: 총 {schedule_labels.family.nunique():,}개 기준 계약·{len(schedule_labels):,}개 시나리오. 모든 계약에 {schedule_paths:,}경로 × {schedule_reps}시드. 위 모델을 재학습 없이 평가했다. 이미 살펴본 시험군의 진단 확장이므로 새 독립 최종시험이라고 부르지 않는다.

### 조건별 MC 가격 변화

계약별 증분의 단순평균이다. 가격 변화량과 예측 오차는 서로 다른 수치다.

{table(['변경 조건','일반형 평균 ΔMC','월지급형 평균 ΔMC'],rows)}

1차 행사가를 바꾸면 원금 회수 시점과 추가 쿠폰 지급이 함께 달라질 수 있다. 가격 변화의 방향과 크기는 해당 계약의 MC 계산값으로 확인한다.

### DeepONet의 가격·증분 예측 성능

아래 가격 지표는 시험 계약의 기준·변경 가격 전체, 증분 지표는 변경 시나리오에서 계산했다. 각 행은 해당 방식의 {len(seeds)}개 학습 시드 평균 예측이다. 방향 일치율은 |ΔMC| > 1.96 × paired MC SE이면서 영향 경로가 30개 이상인 경우만 평가했다.

{table(['상품','학습 방식','가격 R²','가격 MAE(원)','증분 MAE(원)','MC 방향 일치율'],modelrows)}

같은 합성 데이터에서 가격만 학습한 경우와 선택 모델의 평균 증분 오차를 비교하면, {' '.join(improvement)} {ranking_text} 선택은 전체 검증 점수로 결정했으며 시험 결과에 따라 바꾸지 않았다. 이 실험에는 새로운 데이터의 기준 가격만 학습한 별도 대조군이 없어, 합성 자료 추가 효과를 손실 변경 효과와 독립적으로 입증한 것은 아니다.

### 관측 횟수·만기 변경에 대한 추가 평가

관측 +1회의 위치별 정의와 만기 변경에 따른 지급·이자기간 규칙은 [실험 과정](docs/EXPERIMENT.md)에 설명했다. 관측 +1회는 일반형 {reg_obs}개·월지급형 {mon_obs}개에서 평가했다. 공시 계약은 별도 진단 결과 파일에 포함되어 있다.

{table(['변경 조건','일반형 평균 ΔMC(원)','일반형 증분 MAE(원)','월지급형 평균 ΔMC(원)','월지급형 증분 MAE(원)'],sr)}

![관측 횟수·만기 변경의 MC 증분과 선택 모델 예측](docs/assets/schedule_counterfactuals.png)

### 결론과 계약별 오차

{verdict} 평균 증분 MAE의 변화와 조건별 방향·크기 오차를 함께 평가해야 한다.

- 공시 확인 월지급형 32410의 1차 행사가 +5%p: MC **{signed(case.mc_delta_krw)}원**, 선택 DeepONet **{signed(case.pred_delta_krw)}원**으로 방향이 {direction}. 독립 NumPy MC 증분은 **{signed(check.delta_krw)}원**이었다.
- 월지급형 관측 +1회 — {'; '.join(schedule_notes)}.
- 일정 추가 실험은 동일 가중치의 새로운 평가 축이다. 앞의 월지급형 증분 MAE {selected['monthly_test'].delta_mae:.2f}원과 직접 비교해 학습 후 성능이 퇴보했다고 해석하지 않는다. 기존 학습 범위를 벗어나는 일정에서의 예측 오차를 평가한 것이며, 분포 부족만이 오차의 유일한 원인이라고 입증한 것은 아니다.
- 독립 현금흐름 계산·입력/정규화·모델 재추론을 점검했지만 산업용 pricer 전체 검증을 완료했다는 뜻은 아니다. 현재 GBM·역사 변동성/상관·q=0·일별 달력 격자 가정에 따른 모형 오차가 남는다.
- 선택 모델의 가격 R²는 일반형 {selected['regular_test'].price_r2:.4f}, 월지급형 {selected['monthly_test'].price_r2:.4f}였다. 1/5/10원 이내 비율은 증분 예측 오차를 평가하기 위한 진단 지표다. 방향 일치나 평균 오차 하나만으로 모든 계약조건의 증분 안정성을 확정하지 않는다.
'''
    return text, selected_schedule


def figure(predictions, path):
    """Use the same selected-model rows as the commentary, not an old image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for j, (cohort, title) in enumerate([('regular_test_128', 'Regular'), ('monthly_test_128', 'Monthly')]):
        p = select(predictions, cohort=cohort)
        for i, (names, xlabel) in enumerate([
            (['nobs_early', 'nobs_middle', 'nobs_late'], 'Added observation position'),
            (['tenor_months'], 'Maturity change (months)')]):
            q = p[p.axis.isin(names)]
            group = 'axis' if i == 0 else 'offset'
            mean = q.groupby(group)[['mc_delta_krw', 'pred_delta_krw']].mean()
            mean = mean.reindex(names) if i == 0 else mean.sort_index()
            x = np.arange(len(mean)) if i == 0 else mean.index.to_numpy()
            ax = axes[i, j]
            ax.plot(x, mean.mc_delta_krw, 'o-', label='MC')
            ax.plot(x, mean.pred_delta_krw, 's--', label='Selected DeepONet')
            if i == 0:
                ax.set_xticks(x, ['Early', 'Middle', 'Late'])
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set(title=title, xlabel=xlabel, ylabel='Mean price increment (KRW)')
            ax.grid(alpha=.25)
            ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(root=ROOT):
    root = Path(root)
    text, plot_rows = render(root)
    figure(plot_rows, root / 'docs/assets/schedule_counterfactuals.png')
    (root / 'docs/RESULTS.md').write_text(text.replace('](docs/', ']('))
    readme = root / 'README.md'
    if readme.exists():
        prefix = readme.read_text().split('<!-- RESULTS -->')[0]
        readme.write_text(prefix + '<!-- RESULTS -->\n\n' + text)
    experiment = root / 'docs/EXPERIMENT.md'
    if experiment.exists():
        methods = experiment.read_text().replace('# 최종 실험 과정', '# Report 2 — 최종 MC·DeepONet·반사실 실험', 1).replace('](RESULTS.md)', '](docs/RESULTS.md)').replace('](../provenance/', '](provenance/')
        (root / 'Report_2.md').write_text(methods + '\n\n' + text)
    print('RESULTS regenerated from recorded selection, MC labels and predictions')


if __name__ == '__main__':
    main()
