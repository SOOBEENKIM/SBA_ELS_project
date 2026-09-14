"""Clarify inherited descriptive fields without changing any numerical settings."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
changes=[]
for payoff in ['reference','detailed']:
    path=HERE/'synthetic'/payoff/'protocol.json'
    protocol=json.loads(path.read_text())
    text=(f'Nine new Stage-1 checkpoints are trained on this experiment\'s {payoff} payoff labels '
          'with source ATM IV/HV fallback and piecewise-linear zero rates. Three prespecified '
          'learning arms each use three training seeds. They are frozen by validation before '
          'terms and schedule test evaluation. The earlier V3 affine-coupon ensemble is also '
          'evaluated unchanged as a separate prior-frozen comparator. No Stage 2 or PI loss is used.')
    if protocol.get('frozen_model')!=text:
        changes.append(dict(payoff=payoff,field='frozen_model',previous=protocol.get('frozen_model'),current=text))
        protocol['frozen_model']=text
        path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False))
if changes:
    (HERE/'results/metadata_clarification.json').write_text(json.dumps(dict(
        changes=changes,numerical_settings_changed=False,
        reason='Inherited V3 prose described only detailed labels; clarify both current payoffs and the prior-frozen comparator.'),ensure_ascii=False,indent=2))
print('Descriptive metadata fields clarified:',len(changes))
