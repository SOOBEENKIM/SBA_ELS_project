"""Independent checks of the final actual-product FAIR / IV-MC join."""
import csv,gzip,hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'

def main():
    cfg=json.loads((HERE/'protocol.json').read_text())
    for relative,h in cfg['preserved_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==h,relative
    groups=json.loads(gzip.decompress((OUT/'groups.json.gz').read_bytes()))
    records={r['item']:r for g in groups for r in g['records']}
    with (ROOT/'analysis/iv_validation_20260914/results/universe_iv_coverage.csv').open() as f:
        coverage=list(csv.DictReader(f))
    fresh={r['item'] for r in coverage if r['full_iv']=='True'}
    with (OUT/'excluded_products.csv').open() as f:excluded=list(csv.DictReader(f))
    assert set(records)<=fresh and not set(records)&{r['item'] for r in excluded}
    assert len(records)+len(excluded)==len(coverage)==cfg['universe']
    with (OUT/'fair_mc_iv.csv').open() as f:prices=list(csv.DictReader(f))
    assert len(prices)==len(records)==len({r['item'] for r in prices})
    assert {r['item'] for r in prices}==set(records)
    maximum=0.
    for r in prices:
        a=float(r['fair_raw_krw']);b=float(r['mc_face_krw']);issue=float(r['issue_price_krw'])
        err=abs(float(r['gap_krw'])-(a-b)/issue*10000)
        ape_err=abs(float(r['ape_pct'])-abs(a-b)/a*100)
        maximum=max(maximum,err,ape_err);assert max(err,ape_err)<1e-8
        assert int(r['paths_per_seed'])==40000 and int(r['replicates'])==3 and int(r['total_paths'])==120000
    with (ROOT/'analysis/iv_validation_20260914/results/mc_labels.csv').open() as f:
        old={r['family'][len('published_'):]:r for r in csv.DictReader(f)
             if r['origin']=='published_monthly' and r['axis']=='base' and r['arm']=='IV'}
    reference=[]
    for r in prices:
        if r['item'] not in old:continue
        o=old[r['item']];diff=float(r['mc_face_krw'])-float(o['price_krw'])
        z=diff/math.hypot(float(r['mc_se_face_krw']),float(o['price_se_krw']))
        assert abs(z)<6,(r['item'],z)
        reference.append(dict(item=r['item'],price_difference_krw=diff,combined_standard_errors=z))
    assert len(reference)==6
    with (OUT/'price_segments.csv').open() as f:segments=list(csv.DictReader(f))
    assert len(segments)==6 and sum(int(r['n']) for r in segments)==len(prices)
    weighted=sum(float(r['MAPE_pct'])*int(r['n']) for r in segments)/len(prices)
    mean=sum(float(r['ape_pct']) for r in prices)/len(prices)
    assert abs(weighted-mean)<1e-9
    result=dict(status='pass',universe=len(coverage),full_iv=len(fresh),included=len(prices),
        no_iv=len(coverage)-len(fresh),contract_exclusions_after_iv=len(fresh)-len(records),
        max_normalization_identity_error=maximum,six_price_bins_account_for_every_product=True,
        legacy_iv_prices_consistent_with_independent_MC_noise=reference,preserved_inputs_unchanged=True)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
