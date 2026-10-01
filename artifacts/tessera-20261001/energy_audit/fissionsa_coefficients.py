"""Read historical FissionSA coefficients and derive explicit per-bit values."""
import ast
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[4] / 'FissionSA'


def constants(path, wanted):
    def number(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return number(node.left) + number(node.right)
        return Decimal(str(ast.literal_eval(node)))
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = number(node.value)
    assert set(result) == set(wanted)
    return result


def main():
    edp_source = BASE / 'fissionsa/modes/combo6_edp.py'
    macro_source = BASE / 'multitenant/profile_nn_tables.py'
    config_source = BASE / 'Tessera-revision/experiments/ad_cycle/config.json'
    values = constants(edp_source, ('E_READ_PJ', 'E_WRITE_PJ', 'E_RED_MEM_PJ'))
    macro = constants(macro_source, ('E_SRAM_READ_PJ', 'E_SRAM_WRITE_PJ'))
    config = json.loads(config_source.read_text())['memory']
    # Source: multitenant/profile_nn_tables.py module docstring and
    # workloads/workloads-Qwen-models/README.md, SRAM dynamic access section.
    assert '128-bit macro = 8 bf16 words' in macro_source.read_text()
    records = []
    for action, key, bytes_key in (('read', 'E_READ_PJ', 'operand_bytes'),
                                    ('write', 'E_WRITE_PJ', 'accumulator_bytes')):
        bits = config[bytes_key] * 8
        records.append(dict(model='paper/revision modeled-word energy', action=action,
            original_pj=str(values[key]), original_unit='modeled word', equivalent_bits=bits,
            derived_pj_per_bit=str(values[key] / Decimal(bits)), source=str(edp_source),
            note='Per-bit equivalent using revision byte accounting; inherited word coefficients are not per-bit macro measurements.'))
    for action, key in (('read', 'E_SRAM_READ_PJ'), ('write', 'E_SRAM_WRITE_PJ')):
        bits = 128
        records.append(dict(model='older multitenant/Qwen macro energy', action=action,
            original_pj=str(macro[key]), original_unit='128-bit access', equivalent_bits=bits,
            derived_pj_per_bit=str(macro[key] / Decimal(bits)), source=str(macro_source),
            note='Recorded CACTI 32-nm 4-KB 128-bit macro estimate; fractional full-width access convention.'))
    output = HERE / 'fissionsa_sram_coefficients.csv'
    with output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with output.open() as handle:
        back = list(csv.DictReader(handle))
    assert back == [{k: str(v) for k, v in r.items()} for r in records]
    for row in back:
        assert Decimal(row['derived_pj_per_bit']) * int(row['equivalent_bits']) == Decimal(row['original_pj'])
    report = dict(status='PASS', coefficients=records,
        separate_reduction_memory_pj_per_output_word=str(values['E_RED_MEM_PJ']),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), edp_source, macro_source, config_source)})
    with (HERE / 'fissionsa_sram_coefficients.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    assert json.loads((HERE / 'fissionsa_sram_coefficients.json').read_text()) == report
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
