#!/usr/bin/env python3
"""Run the full parameter/mode matrix against a clean clone of this repository.

Safe by design: every search is time-limited (`--seconds`) or uses a tiny slice;
`--ladder` only solves puzzles 1..5; `--exhaust` uses tiny slices; `--freeze --vis`
is CPU-only. The clone keeps all runtime files (found.txt, freeze masks, exhaust
results) out of the real repository.
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix='btcpuzzle-snake-matrix-'))
WORK = TMP / 'repo'
DATA = TMP / 'data'
REPORT = REPO / 'docs' / 'MODE_MATRIX.md'


def find_cuda():
    for d in [os.path.expanduser('~/cudaroot/usr/local/cuda-12.8'),
              os.path.expanduser('~/cudaroot/usr/local/cuda'),
              '/usr/local/cuda-12.8', '/usr/local/cuda', '/opt/cuda']:
        if Path(d, 'bin', 'nvcc').exists():
            return d
    return None


def sh(cmd, cwd, timeout=600):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    out = (p.stdout + p.stderr).decode('utf-8', 'replace')
    return p.returncode, out


def main():
    if not REPO.exists():
        print('repo not found:', REPO)
        return 2
    cuda_home = find_cuda()
    if not cuda_home:
        print('CUDA toolkit not found')
        return 2
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f'{cuda_home}/lib64:/usr/lib/wsl/lib:' + env.get('LD_LIBRARY_PATH', '')

    if WORK.exists():
        shutil.rmtree(WORK)
    if DATA.exists():
        shutil.rmtree(DATA)
    print('=== cloning', REPO, '->', WORK)
    rc, out = sh(['git', 'clone', '-q', str(REPO), str(WORK)], WORK.parent, timeout=120)
    if rc != 0:
        print('clone failed:\n', out)
        return 2
    commit = sh(['git', 'rev-parse', '--short', 'HEAD'], WORK)[1].strip()

    print('=== building snake and scan_range')
    build1 = sh(['bash', 'cuda/build.sh'], WORK, timeout=600)
    build2 = sh(['bash', 'cuda/build_scan.sh'], WORK, timeout=600)
    if build1[0] != 0 or build2[0] != 0:
        print('build failed')
        print('snake:', build1[1][-2000:])
        print('scan:', build2[1][-2000:])
        return 2

    CUDA = WORK / 'cuda'
    BIN = {'snake': CUDA / 'snake', 'scan': CUDA / 'scan_range'}
    DATA.mkdir(parents=True, exist_ok=True)

    all_lines = (WORK / 'allPuzzles.js').read_text(encoding='utf-8').splitlines()
    if len(all_lines) < 160:
        print('allPuzzles.js has', len(all_lines), 'lines')
        return 2
    addr16 = all_lines[15].strip()
    short_path = DATA / 'short.js'
    short_path.write_text('\n'.join(all_lines[:100]) + '\n', encoding='utf-8')
    bad_lines = all_lines[:160]
    bad_lines[70] = 'xxx'
    bad_path = DATA / 'bad.js'
    bad_path.write_text('\n'.join(bad_lines) + '\n', encoding='utf-8')
    hash40 = '87f0201511180e729bed2d80da1880c3901f1199'  # hash160 of key 12345

    cases = []

    def add(tool, args, expect='ok', marker=None, timeout=30, note=''):
        cases.append({'tool': tool, 'args': args, 'expect': expect,
                      'marker': marker, 'timeout': timeout, 'note': note})

    # ---------------- CLI / help / errors ----------------
    add('snake', ['--help'], 'ok', 'Modes:', 20, 'all modes listed')
    add('snake', ['-h'], 'ok', 'Modes:', 20)
    add('snake', ['--bogus'], 'fail', 'Usage:', 20)
    add('snake', [], 'timeout', 'Snake:', 3, 'no args = default snake (killed after 3 s)')
    for opt in ['--spread', '--exhaust', '--freeze', '--kmin', '--kmax', '--frz',
                '--from', '--to', '--ref', '--ones', '--chunkfrom', '--load',
                '--seconds', '--addr', '--show']:
        add('snake', [opt], 'fail', 'Usage:', 15, 'missing value')
    add('snake', ['--snake', '--spread', '71'], 'fail', 'do not combine', 15)
    add('snake', ['--snake', '--exhaust', '16'], 'fail', 'do not combine', 15)
    add('snake', ['--snake', '--freeze', '16'], 'fail', 'do not combine', 15)
    add('snake', ['--snake', '--ladder'], 'fail', 'do not combine', 15)
    add('snake', ['--vis'], 'fail', '--vis is a picture', 15)
    add('snake', ['--snake', '--vis'], 'fail', '--vis is a picture', 15)

    # ---------------- default snake / common options ----------------
    add('snake', ['--snake', '--seconds', '1'], 'ok', 'Stopped:')
    add('snake', ['--snake', '--seconds', '0.5'], 'ok', 'Stopped:')
    add('snake', ['--snake', '--seconds', '1', '--nocool'], 'ok', 'Stopped:')
    add('snake', ['--snake', '--seconds', '1', '--load', '5'], 'ok', 'load limit 5%')
    add('snake', ['--snake', '--seconds', '1', '--load', '1'], 'ok', 'load limit 5%', 30, 'clamps to 5')
    add('snake', ['--snake', '--seconds', '1', '--load', '200'], 'ok', 'load limit off', 30, 'clamps to 100')
    add('snake', ['--snake', '--seconds', '1', '--load', 'abc'], 'ok', 'load limit 5%', 30, 'atoi(0) -> 5')
    add('snake', ['--snake', '--seconds', '1', '--load', '-5'], 'ok', 'load limit 5%', 30, 'clamps to 5')
    add('snake', ['--snake', '--seconds', '1', '--addr', '../allPuzzles.js'], 'ok', 'Snake:')
    add('snake', ['--snake', '--seconds', '1', '--addr', str(WORK / 'allPuzzles.js')], 'ok', 'Snake:')
    add('snake', ['--snake', '--addr', '/nonexistent'], 'fail', 'cannot read 160 addresses', 20)
    add('snake', ['--snake', '--addr', str(short_path)], 'fail', 'cannot read 160 addresses', 20, '100 lines')
    add('snake', ['--snake', '--addr', str(bad_path)], 'fail', 'CPU self-test FAILED', 20, 'bad Base58 at line 71')
    add('snake', ['--snake', '--seconds', '1', '--watch', '--ref', '3', '--ones', '3'], 'ok', 'РЕШЕНО')
    add('snake', ['--snake', '--seconds', '1', '--watch', '--ref', '16', '--ones', '8'], 'ok', 'watch:')
    add('snake', ['--snake', '--seconds', '1', '--watch', '--ref', '70'], 'ok', 'watch:')
    add('snake', ['--snake', '--seconds', '1', '--watch', '--ref', '71'], 'fail', 'нет известного ключа')
    add('snake', ['--snake', '--seconds', '1', '--check'], 'ok', 'Stopped:', 30, 'ignored outside freeze')
    add('snake', ['--snake', '--seconds', '1', '--kmin', '5', '--kmax', '5'], 'ok', 'Stopped:', 30, 'ignored in snake')
    add('snake', ['--snake', '--seconds', '1', '--chunkfrom', '5'], 'ok', 'Stopped:', 30, 'ignored in snake')
    add('snake', ['--snake', '--seconds', '1', '--frz', '5'], 'ok', 'Stopped:', 30, 'ignored in snake')
    add('snake', ['--snake', '--seconds', '1', '--from', '1', '--to', '2'], 'ok', 'Stopped:', 30, 'ignored in snake')
    add('snake', ['--show', '1'], 'ok', 'snake step 0')
    add('snake', ['--show', '2'], 'ok', 'snake step 1')
    add('snake', ['--show', '0', '--seconds', '1'], 'ok', 'Snake:', 30, 'show 0 = normal snake')
    add('snake', ['--show', '-1', '--seconds', '1'], 'ok', 'Snake:', 30, 'show -1 = normal snake')
    add('snake', ['--test'], 'ok', 'GPU self-test PASSED')
    add('snake', ['--snake', '--test'], 'ok', 'GPU self-test PASSED')
    add('snake', ['--alert-test'], 'ok', 'alert.py failed', 20, 'repo has no alert.py')

    # ---------------- spread ----------------
    add('snake', ['--spread', '16', '--kmin', '2', '--kmax', '2', '--seconds', '1'], 'ok', 'Spread: puzzle 16')
    add('snake', ['--spread', '70', '--kmin', '2', '--kmax', '2', '--seconds', '1'], 'ok', 'Spread: puzzle 70')
    add('snake', ['--spread', '71', '--seconds', '1'], 'ok', 'Spread: puzzle 71')
    add('snake', ['--spread', '160', '--kmin', '2', '--kmax', '2', '--seconds', '1'], 'ok', 'Spread: puzzle 160')
    add('snake', ['--spread', '160', '--kmin', '160', '--kmax', '160', '--seconds', '1'], 'ok', 'ones 159..159')
    add('snake', ['--spread', '16', '--kmin', '1', '--kmax', '1', '--seconds', '1'], 'ok', 'ones 2..2')
    add('snake', ['--spread', '16', '--kmin', '20', '--kmax', '30', '--seconds', '1'], 'ok', 'ones 15..15')
    add('snake', ['--spread', '16', '--kmin', '5', '--kmax', '3', '--seconds', '1'], 'ok', 'ones 5..5')
    add('snake', ['--spread', '16', '--kmin', '2', '--kmax', '2', '--load', '5', '--seconds', '1'], 'ok', 'load limit 5%')
    add('snake', ['--spread', '16', '--kmin', '2', '--kmax', '2', '--nocool', '--seconds', '1'], 'ok', 'Spread: puzzle 16')
    add('snake', ['--spread', '3', '--watch', '--ref', '3', '--ones', '3', '--seconds', '1'], 'ok', 'Watch:', 30, 'spread watch shows the walker; --ones is not used here')
    add('snake', ['--spread', '16', '--watch', '--ref', '16', '--ones', '8', '--seconds', '1'], 'ok', 'Watch:')
    add('snake', ['--spread', '16', '--watch', '--ref', '16', '--ones', '0', '--seconds', '1'], 'ok', 'Watch:')
    add('snake', ['--spread', '16', '--watch', '--ref', '70', '--ones', '8', '--seconds', '1'], 'ok', 'Watch:')
    add('snake', ['--spread', '2'], 'fail', 'puzzle 3..160', 20)
    add('snake', ['--spread', '161'], 'fail', 'puzzle 3..160', 20)
    add('snake', ['--spread', '0'], 'fail', 'puzzle 3..160', 20)

    # ---------------- exhaust ----------------
    add('snake', ['--exhaust', '4', '--kmin', '2', '--kmax', '2'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '16', '--kmin', '2', '--kmax', '2'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '71', '--kmin', '2', '--kmax', '2'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '128', '--kmin', '2', '--kmax', '2'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '3'], 'fail', '4..128', 20)
    add('snake', ['--exhaust', '0'], 'fail', '4..128', 20)
    add('snake', ['--exhaust', '129'], 'fail', '4..128', 20)
    add('snake', ['--exhaust', '130'], 'fail', '4..128', 20)
    add('snake', ['--exhaust', '16', '--kmin', '1', '--kmax', '1'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '16', '--kmin', '16', '--kmax', '16'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '16', '--kmin', '17', '--kmax', '17'], 'ok', 'impossible')
    add('snake', ['--exhaust', '16', '--kmin', '2', '--kmax', '2', '--force'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '16', '--kmin', '2', '--kmax', '2', '--load', '5'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '16', '--kmin', '2', '--kmax', '2', '--seconds', '0.1'], 'ok', 'TOTAL:', 30, 'seconds ignored by exhaust')
    add('snake', ['--exhaust', '71', '--kmin', '36', '--kmax', '36'], 'ok', 'too big')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '7'], 'ok', '27,132 / 27,132')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '8'], 'ok', 'TOTAL:')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '7', '--chunkfrom', '0'], 'ok', '27,132 / 27,132')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '7', '--chunkfrom', '1'], 'ok', '18,940 / 18,940')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '7', '--chunkfrom', '2'], 'ok', '10,748 / 10,748')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '7', '--chunkfrom', '4'], 'ok', 'nothing to do')
    add('snake', ['--exhaust', '20', '--kmin', '7', '--kmax', '7', '--chunkfrom', '99'], 'ok', 'nothing to do')
    add('snake', ['--exhaust', '16', '--kmin', '2', '--kmax', '2', '--chunkfrom', '1'], 'ok', 'nothing to do')

    # ---------------- freeze ----------------
    add('snake', ['--freeze', '16', '--frz', '8', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '1', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '15', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '0', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '8', '--check', '--seconds', '1'], 'ok', 'test hit on solved puzzle 16')
    add('snake', ['--freeze', '3', '--frz', '1', '--check', '--seconds', '1'], 'ok', 'test hit on solved puzzle 3')
    add('snake', ['--freeze', '3', '--frz', '2', '--check', '--seconds', '1'], 'ok', 'test hit on solved puzzle 3')
    add('snake', ['--freeze', '71', '--frz', '50', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '71', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '128', '--frz', '100', '--seconds', '1'], 'ok', 'Freeze:', 40)
    add('snake', ['--freeze', '71', '--frz', '20', '--seconds', '1'], 'fail', 'use --frz')
    add('snake', ['--freeze', '2'], 'fail', '3..128', 20)
    add('snake', ['--freeze', '0'], 'fail', '3..128', 20)
    add('snake', ['--freeze', '129'], 'fail', '3..128', 20)
    add('snake', ['--freeze', '130'], 'fail', '3..128', 20)
    add('snake', ['--freeze', '16', '--frz', '16', '--seconds', '1'], 'fail', '--frz')
    add('snake', ['--freeze', '16', '--frz', '16', '--vis', '--seconds', '1'], 'fail', '--vis: --frz')
    add('snake', ['--freeze', '129', '--vis', '--seconds', '1'], 'fail', '3..128')
    add('snake', ['--freeze', '16', '--frz', '8', '--force', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '8', '--load', '5', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '8', '--watch', '--seconds', '1'], 'ok', 'Freeze:')
    add('snake', ['--freeze', '16', '--frz', '8', '--vis', '--seconds', '1'], 'ok', 'Freeze visualisation', 20)
    add('snake', ['--freeze', '71', '--vis', '--seconds', '1'], 'ok', 'Freeze visualisation', 20)

    # ---------------- ladder ----------------
    add('snake', ['--ladder', '--from', '1', '--to', '5'], 'ok', 'лестница: всего', 40)
    add('snake', ['--ladder', '--from', '1', '--to', '1'], 'ok', 'лестница: всего', 40)
    add('snake', ['--ladder', '--from', '0', '--to', '2'], 'ok', 'решён', 40)
    add('snake', ['--ladder', '--to', '0'], 'ok', 'лестница: всего', 20)
    add('snake', ['--ladder', '--from', '5', '--to', '1'], 'ok', 'лестница: всего', 20)
    add('snake', ['--ladder', '--from', '1', '--to', '5', '--seconds', '1'], 'ok', 'лестница: всего', 40, 'seconds ignored by ladder')
    add('snake', ['--ladder', '--from', '1', '--to', '5', '--load', '5'], 'ok', 'лестница: всего', 40, 'load ignored by ladder')
    add('snake', ['--ladder', '--from', '1', '--to', '5', '--watch'], 'ok', 'лестница: всего', 40)

    # ---------------- scan_range ----------------
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000'], 'ok', 'НАЙДЕНО')
    add('scan', ['--plant', '51510', '--start', '1', '--count', '60000'], 'ok', 'НАЙДЕНО', 30, 'start is a power of two (regression)')
    add('scan', ['--plant', '51510', '--start', '1', '--count', '60000', '--blocks', '2048'], 'ok', 'НАЙДЕНО', 30, 'same with more blocks')
    add('scan', ['--plant', '12345', '--start', '1', '--count', '20000', '--per', '2'], 'ok', 'НАЙДЕНО', 30, 'per>1 + key 1 (regression)')
    add('scan', ['--plant', '1', '--start', '0', '--count', '2'], 'ok', 'НАЙДЕНО', 30, 'start 0 = infinity (regression)')
    add('scan', ['--plant', '12345', '--start', '0x2ee0', '--count', '1000'], 'ok', 'НАЙДЕНО')
    add('scan', ['--target', hash40, '--start', '12000', '--count', '1000'], 'ok', 'НАЙДЕНО')
    add('scan', ['--target', addr16, '--start', '1', '--count', '60000'], 'ok', 'НАЙДЕНО')
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--per', '64', '--blocks', '4'], 'ok', 'НАЙДЕНО')
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--per', '0'], 'ok', 'НАЙДЕНО')
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--blocks', '1', '--per', '1000'], 'ok', 'НАЙДЕНО')
    add('scan', ['--plant', '12345', '--start', '12000'], 'ok', 'НАЙДЕНО', 30, 'default count')
    add('scan', ['--plant', '1180591620717411303424', '--count', '1000000'], 'ok', 'НАЙДЕНО', 30, 'default start 2^70')
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--target', hash40], 'ok', 'plant:')
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '0'], 'fail', 'совпадений 0')
    add('scan', [], 'fail', 'нужен --target или --plant', 20)
    add('scan', ['--target', 'xyz', '--start', '0', '--count', '10'], 'fail', 'не разобрал --target', 20)
    add('scan', ['--plant', 'xyz', '--start', '0', '--count', '10'], 'fail', 'плохое число', 20)
    add('scan', ['--start', 'xyz', '--count', '10', '--plant', '1'], 'fail', 'ожидается десятичное', 20)
    add('scan', ['--blocks', '0', '--plant', '12345', '--start', '12000', '--count', '1000'], 'fail', 'must be a positive number', 20)
    add('scan', ['--blocks', '-1', '--plant', '12345', '--start', '12000', '--count', '1000'], 'fail', 'must be a positive number', 20)
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--per'], 'fail', 'usage:', 20)
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--target'], 'fail', 'usage:', 20)
    add('scan', ['--plant', '12345', '--start', '12000', '--count', '1000', '--blocks'], 'fail', 'usage:', 20)

    # ---------------- run ----------------
    results = []
    t0 = time.time()
    print(f'=== {len(cases)} cases, commit {commit}')
    for i, c in enumerate(cases, 1):
        cmd = ['stdbuf', '-oL', '-eL', str(BIN[c['tool']])] + c['args']
        display = shlex.join([c['tool']] + c['args'])
        display = display.replace(str(WORK), '<clone>').replace(str(DATA), '<tmp>')
        timed_out = False
        try:
            p = subprocess.run(cmd, cwd=CUDA, env=env, capture_output=True, timeout=c['timeout'])
            rc = p.returncode
            out = (p.stdout + p.stderr).decode('utf-8', 'replace')
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = None
            out = ((e.stdout or b'') + (e.stderr or b'')).decode('utf-8', 'replace')
        if c['expect'] == 'timeout':
            ok = timed_out
        elif c['expect'] == 'ok':
            ok = (not timed_out and rc == 0)
        else:
            ok = (not timed_out and rc != 0)
        marker_ok = (c['marker'] is None) or (c['marker'] in out)
        ok = ok and marker_ok
        results.append({'case': c, 'display': display, 'rc': rc, 'out': out,
                        'timed_out': timed_out, 'ok': ok, 'marker_ok': marker_ok})
        print(f"[{i:03d}/{len(cases)}] {'PASS' if ok else 'FAIL'}  {display}")

    elapsed = time.time() - t0
    # Check that solved-puzzle test hits did not leak into found.txt.
    found = CUDA / 'found.txt'
    leaked = False
    found_text = ''
    if found.exists():
        found_text = found.read_text(encoding='utf-8', errors='replace')
        leaked = ('puzzle   16' in found_text) or ('puzzle   3' in found_text)
    no_leak_ok = not leaked
    results.append({'case': {'tool': 'check', 'args': [], 'expect': 'ok',
                             'marker': None, 'timeout': 0, 'note': 'found.txt must stay clean'},
                    'display': 'check: solved-puzzle test hits are not written to found.txt',
                    'rc': 0 if no_leak_ok else 1, 'out': found_text[-500:],
                    'timed_out': False, 'ok': no_leak_ok, 'marker_ok': True})

    passed = sum(1 for r in results if r['ok'])
    failed = len(results) - passed
    gpu = sh(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], WORK, timeout=20)[1].strip()
    date = time.strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    lines.append(f'# btcpuzzle_snake_gpu mode matrix — {date}\n')
    lines.append(f'Clean clone of this repository at commit **`{commit}`**.\n')
    lines.append(f'- GPU: {gpu or "unknown"}')
    lines.append(f'- CUDA: `{cuda_home.replace(str(Path.home()), "~")}`')
    lines.append(f'- Build: `cuda/build.sh` + `cuda/build_scan.sh` — OK')
    lines.append(f'- Cases: **{len(results)}**, passed **{passed}**, failed **{failed}**, wall time {elapsed:.1f} s')
    lines.append('- The clone is temporary; runtime files (`found.txt`, freeze masks, `exhaust_results.txt`) do not touch the real repository.\n')
    lines.append('## Matrix\n')
    lines.append('| # | case | command | expected | rc | verdict | note |')
    lines.append('|---:|---|---|---|---:|---|---|')
    for i, r in enumerate(results, 1):
        c = r['case']
        exp = {'ok': 'rc=0', 'fail': 'rc!=0', 'timeout': 'timeout'}[c['expect']]
        rc = 'timeout' if r['timed_out'] else str(r['rc'])
        note = c['note']
        if not r['marker_ok'] and c['marker']:
            note = (note + '; ' if note else '') + f'missing marker: {c["marker"]}'
        if not r['ok'] and r['ok'] is False:
            note = (note + '; ' if note else '') + 'FAILED'
        cmd = r['display'].replace('|', '\\|')
        note = (note or '').replace('|', '\\|')
        lines.append(f'| {i} | {c["tool"]} | `{cmd}` | {exp} | {rc} | {"PASS" if r["ok"] else "FAIL"} | {note} |')

    if failed:
        lines.append('\n## Failures (tail of output)\n')
        for i, r in enumerate(results, 1):
            if r['ok']:
                continue
            c = r['case']
            lines.append(f'### {i}. `{r["display"]}` — rc {r["rc"]}')
            lines.append('```')
            tail = [l for l in r['out'].splitlines() if l.strip()][-12:]
            lines.extend(tail)
            lines.append('```\n')
    else:
        lines.append('\n## Result\n\nAll cases passed.\n')

    lines.append('## Notes / observed behaviour\n')
    lines.append('- `--seconds` limits the default snake, `--spread`, `--freeze` and `--freeze --vis`; '
                 'it is ignored by `--exhaust` and `--ladder` (they run their unit of work to the end).')
    lines.append('- `--load` affects the default snake, `--spread`, `--freeze` and `--exhaust` duty cycles; '
                 '`--ladder` ignores it.')
    lines.append('- `--exhaust --chunkfrom` only applies when `kmin == kmax` (one slice); a resumed run now '
                 'reports the keys checked in this run and marks the slice as resumed.')
    lines.append('- `--exhaust` supports puzzles 4..128 and `--freeze` 3..128: the top bit is kept outside the '
                 '128-bit mask, so puzzle 129 cannot be represented.')
    lines.append('- `--freeze --check` caps the frozen count to the known key\'s popcount; test hits on solved '
                 'puzzles are printed but not appended to `found.txt`.')
    lines.append('- `scan_range` accepts a Base58 address or a 40-hex hash160 in `--target`, and returns rc 1 '
                 'when the requested range contains no match (including `--count 0`).')
    lines.append('- `--alert-test` returns 0 even though the repo intentionally has no `alert.py`; it prints '
                 '`alert.py failed`.')
    lines.append('- `--show 0` and negative values fall through to the normal snake search.')
    lines.append('- `--snake --test` runs the self-tests (test mode wins over the default search).')
    lines.append('- Known-puzzle test hits (`--freeze --check`, `--spread --watch`, `--snake --watch`) do not '
                 'pollute `found.txt`; the clone check confirmed it.')
    lines.append('')
    lines.append('## Bugs found and fixed in this tree\n')
    lines.append('1. `scan_range --blocks 0` died with SIGFPE (division by zero); `--blocks -1` reached CUDA '
                 'and failed with `invalid configuration argument`. Both now return a clear rc 1 before CUDA.')
    lines.append('2. `--exhaust --chunkfrom` reported the whole slice as checked even when it started in the '
                 'middle (e.g. `--chunkfrom 4` with 4 chunks reported `27,132 / 27,132`). It now reports the '
                 'keys actually checked, the skipped prefix, and `nothing to do` when the start is past the end.')
    lines.append('3. `--exhaust` advertised 3..129 but the GPU unranking table was sized for n<=70: puzzle 72 '
                 'was undefined behaviour and 129 could not encode its top bit. The table now covers n<=128, '
                 'the small-nb self-test is fixed, and the valid range is 4..128 (freeze/vis: 3..128).')
    lines.append('4. `--freeze --vis --frz` larger than the low-bit count was silently clamped on solved '
                 'puzzles; it now rejects it like the search mode does.')
    lines.append('5. `--exhaust` and `--freeze` appended solved-puzzle test hits to `found.txt`; they no longer '
                 'do that (real finds still go there).')
    lines.append('6. `scan_range` missed a target when the range started at a power of two and the per-thread '
                 'base offset had that bit set (e.g. `--plant 51510 --start 1 --count 60000` returned 0 hits). '
                 '`madd()` assumes distinct points, so `P0 + P0` divided by zero. A scanner-specific '
                 '`madd_scan()` now doubles on equality and handles infinity; verified with `start=1` and `start=0`.')
    lines.append('7. `tools/gen_scan_cu.py` sliced `snake.cu` by hardcoded line numbers. Adding four lines to '
                 '`fe_eq` silently produced a broken `scan_range.cu` (`__constant__` inside a host function). '
                 'The generator now uses `GEN_SCAN_*` marker comments; the generated scanner is built and '
                 'exercised in this matrix.')

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n=== {passed}/{len(results)} passed, {failed} failed, {elapsed:.1f}s')
    print('report:', REPORT)
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
