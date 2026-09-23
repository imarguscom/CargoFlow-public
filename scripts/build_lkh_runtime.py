#!/usr/bin/env python3
"""Download and build LKH 3.0.13 locally, with the directed-matrix safety patch."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'LKH-3.0.13'
URL = f'http://webhotel4.ruc.dk/~keld/research/LKH-3/{VERSION}.tgz'
PATCH = '/* CargoFlow: disable metric-only TSPTW precedence reduction. */'


def extract_source(archive, destination):
    """Extract regular source files only, contained in the expected directory."""
    with tarfile.open(archive) as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != VERSION:
                raise ValueError('Unexpected path in LKH archive')
            target = destination.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            else:
                raise ValueError('LKH archive contains a non-regular file')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use a locally downloaded author archive')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/runtime_lkh')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    destination = args.output_dir.resolve()
    source = destination/VERSION
    if source.exists():
        if (source/'LKH').is_file() and PATCH in (source/'SRC/LKHmain.c').read_text():
            print('Patched LKH runtime is ready.')
            return
        parser.error('Output already exists; choose a fresh --output-dir')
    if not shutil.which('make'):
        parser.error('Install make and a C compiler first')
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-', dir=destination) as temp:
        work = Path(temp)
        archive = args.archive.resolve() if args.archive else work/(VERSION+'.tgz')
        if args.archive is None:
            if not shutil.which('curl'):
                parser.error('Install curl or supply --archive')
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--connect-timeout', '20', '--max-time', '300', '--retry', '2',
                            URL, '--output', str(archive)], check=True)
        extract_source(archive, work)
        build = work/VERSION
        main_c = build/'SRC/LKHmain.c'
        text = main_c.read_text()
        original = '    if (ProblemType == TSPTW)\n        TSPTW_Reduce();'
        if text.count(original) != 1:
            raise ValueError('Unexpected LKH source; directed-matrix patch was not applied')
        main_c.write_text(text.replace(original, '    '+PATCH))
        log = work/'build.log'
        with log.open('w') as output:
            result = subprocess.run(['make', '-C', str(build/'SRC'), f'-j{args.jobs}'],
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-6000:])
            raise RuntimeError('LKH compilation failed')
        if not (build/'LKH').is_file():
            raise RuntimeError('LKH build did not produce an executable')
        build.rename(source)
    print('LKH 3.0.13 built with the directed-matrix patch.')


if __name__ == '__main__':
    main()
