#!/usr/bin/env python3
"""Verifica CSV contra logs originales y resume exclusivamente las 21 corridas A."""

import argparse
import csv
import json
from pathlib import Path
import statistics

from run_a_serial import BLOCKS, ROUNDS, parse_output, sha, write_json


def verified_rows(campaign):
    """Rechaza registros incompletos, alterados o incongruentes con su salida."""
    metadata = json.loads((campaign / 'metadata.json').read_text())
    if not metadata.get('all_valid') or not metadata['diagnostic']['single_thread_confirmed']:
        raise ValueError(f'Campaña incompleta o sin un hilo confirmado: {campaign}')
    with (campaign / 'raw.csv').open() as f:
        rows = list(csv.DictReader(f))
    expected = [tuple(x) for x in metadata['schedule']]
    actual = [(int(r['repetition']), int(r['N']), int(r['b'])) for r in rows]
    if actual != expected or len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('Orden/cobertura/identificadores distintos del programa registrado')
    if len(list((campaign / 'logs').glob('*.json'))) != len(rows):
        raise ValueError('Existen registros faltantes o inconclusos')
    for order, row in enumerate(rows, 1):
        if (row['status'] != 'valido' or row['returncode'] != '0'
                or int(row['order']) != order or row['type'] != metadata['phase']):
            raise ValueError(f'Corrida inválida: {row["run_id"]}')
        for stream in ['stdout', 'stderr']:
            if sha(campaign / row[stream]) != row[stream + '_sha256']:
                raise ValueError(f'Hash de {stream} distinto: {row["run_id"]}')
        if (campaign / row['stderr']).stat().st_size:
            raise ValueError('stderr no vacío')
        t, residual = parse_output((campaign / row['stdout']).read_text(),
                                  int(row['N']), int(row['b']), row['type'] == 'diagnostico')
        if t != float(row['factorization_s']):
            raise ValueError('El tiempo CSV no reproduce stdout')
        if residual is not None and residual != float(row['residual']):
            raise ValueError('El residuo CSV no reproduce stdout')
        record = json.loads((campaign / 'logs' / (row['run_id'] + '.json')).read_text())
        if any(str(record['row'].get(k, '')) != v for k, v in row.items()
               if not (k == 'residual' and v == '' and record['row'].get(k) is None)):
            raise ValueError('Registro individual distinto del CSV')
    return rows


def summarize(campaign):
    rows = verified_rows(campaign)
    expected = [(rep, 4096, b) for rep, order in enumerate(ROUNDS, 1) for b in order]
    if (len(rows) != 21 or any(r['type'] != 'oficial' for r in rows)
            or [(int(r['repetition']), int(r['N']), int(r['b'])) for r in rows] != expected):
        raise ValueError('Se requieren exactamente las 21 corridas oficiales del protocolo')
    result = []
    for b in sorted(BLOCKS):
        times = [float(r['factorization_s']) for r in rows if int(r['b']) == b]
        median = statistics.median(times)
        result.append(dict(b=b, blocks=4096 // b, valid_repetitions=len(times),
                           t1_s=times[0], t2_s=times[1], t3_s=times[2],
                           median_s=median, min_s=min(times), max_s=max(times),
                           range_s=round(max(times) - min(times), 3),
                           gflops_model=(2 * 4096**3 / 3) / median / 1e9))
    return result


def table(summary):
    best = min(summary, key=lambda r: r['median_s'])['b']
    lines = ['| b | N/b | t₁ (s) | t₂ (s) | t₃ (s) | Mediana (s) | Mín–máx (s) | Rango (s) | GFLOP/s¹ |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        label = f'**{r["b"]}***' if r['b'] == best else str(r['b'])
        lines.append(f'| {label} | {r["blocks"]} | {r["t1_s"]:.3f} | {r["t2_s"]:.3f} | '
                     f'{r["t3_s"]:.3f} | {r["median_s"]:.3f} | '
                     f'{r["min_s"]:.3f}–{r["max_s"]:.3f} | {r["range_s"]:.3f} | '
                     f'{r["gflops_model"]:.2f} |')
    lines += ['', '* Menor mediana observada; propuesta de b_A pendiente de aprobación.',
              '¹ Modelo (2 N³/3)/mediana/10⁹; no es un contador de FLOPs de hardware.',
              'N=4096; tres procesos independientes por bloque; un hilo; tiempos internos impresos con .3f.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Directorio nuevo')
    args = parser.parse_args()
    summary = summarize(args.campaign)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'summary.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    ranked = sorted(summary, key=lambda r: r['median_s'])
    write_json(args.output / 'summary.json', dict(
        raw_sha256=sha(args.campaign / 'raw.csv'), rows=summary,
        proposed_b_A=ranked[0]['b'], next_b=ranked[1]['b'],
        next_median_gap_s=ranked[1]['median_s'] - ranked[0]['median_s'],
        next_median_gap_percent=100 * (ranked[1]['median_s'] / ranked[0]['median_s'] - 1)))
    (args.output / 'tabla_a.md').write_text(table(summary))
    print(table(summary))


if __name__ == '__main__':
    main()
