"""Copy only frozen sources and executable analysis into a new sibling run."""
from pathlib import Path
import argparse, shutil

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True,help='New directory name under analysis/')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in {'.','..'}:
        parser.error('Use a single new directory name, without slashes')
    dest=HERE.parent/args.name
    dest.mkdir(exist_ok=False)
    for source in HERE.glob('*.py'):shutil.copy2(source,dest/source.name)
    for name in ['source_manifest.json','EXECUTION.md','.gitignore','.gitattributes']:
        shutil.copy2(HERE/name,dest/name)
    shutil.copytree(HERE/'reference',dest/'reference')
    print(f'Created {dest}; no old labels, models or test results copied.')
    print(f'Run the commands in EXECUTION.md with analysis/{args.name} as the analysis directory.')

if __name__=='__main__':main()
