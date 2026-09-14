"""Restore the attached IV archive without overwriting differing local files."""
from pathlib import Path
import hashlib,json,zipfile

def main():
    root=Path(__file__).resolve().parents[2]
    dest=root/'data/raw/iv_daily_atm'
    spec=json.loads((dest/'source.json').read_text())
    archive=dest/'iv_daily_atm.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==spec['sha256']
    count=0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            p=Path(info.filename)
            assert len(p.parts)==2 and p.parts[0]=='iv_daily_atm' and p.suffix=='.csv'
            data=z.read(info);target=dest/p.name
            if target.exists():assert target.read_bytes()==data,f'Existing IV file differs: {target}'
            else:target.write_bytes(data)
            count+=1
    assert count==spec['csv_count']
    print(f'PASS: IV ZIP SHA-256 and all {count} CSV payloads verified.')

if __name__=='__main__':main()
