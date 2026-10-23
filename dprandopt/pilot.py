"""CPU-only manifest and atomic candidate checkpoints for paired cap pilots."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .countdown import generate_instances, to_messages
from .scoretable import ScoreTable


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def context_budget(prompt_tokens, output_tokens, requested=None):
    """Reserve prompt AND completion tokens; do not silently shorten generation."""
    required = int(prompt_tokens) + int(output_tokens)
    limit = requested if requested is not None else max(2048, 1 << (required-1).bit_length())
    if limit < required:
        raise ValueError('Prompt plus output budget exceeds the pilot context limit')
    return limit


def prepare(reference, count, max_tokens):
    reference = Path(reference)
    table = ScoreTable.load(str(reference))
    meta = json.loads(Path(str(reference) + '.meta.json').read_text())
    if meta['dataset'] != 'countdown' or table.base_index != 0:
        raise ValueError('This paired pilot requires Countdown and base row 0')
    if not 1 <= count <= table.n_candidates or max_tokens < 512:
        raise ValueError('Invalid pilot size or cap; expected at least 512 tokens')
    if table.output_lengths is None or np.max(table.output_lengths) != 512:
        raise ValueError('Reference must contain the archived 512-token lengths')
    # Verify that the stored manifest matches the original FULL-size RNG draw.
    rng = np.random.default_rng(meta['candidate_seed'])
    seeds = np.r_[-1, rng.choice(2**31-1, size=table.n_candidates-1, replace=False)]
    sigmas = np.r_[0., rng.choice(meta['sigmas'], size=table.n_candidates-1)]
    if not np.array_equal(seeds, table.seed) or not np.array_equal(sigmas, table.sigma):
        raise ValueError('Reference candidate seeds/sigmas disagree with original metadata')
    sizes = [meta[k] for k in ['pool','public','test']]
    if sizes != [table.pool_size, table.public_reward.shape[1], table.test_size]:
        raise ValueError('Reference sizes disagree with metadata')
    instances = generate_instances(sum(sizes), seed=meta['item_seed'])
    items = [{'messages':to_messages(x), 'ground_truth':{'numbers':x['numbers'],'target':x['target']},
              'numbers':x['numbers'], 'target':x['target']} for x in instances]
    manifest = dict(schema=1, reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        model=meta['model'], dataset='countdown', pool=sizes[0], public=sizes[1], test=sizes[2],
        item_seed=meta['item_seed'], decoding_seed=meta['candidate_seed'],
        temperature=0., precision='bfloat16', reference_max_tokens=512, max_tokens=max_tokens,
        items_sha256=digest(items), candidate_indices=list(range(count)),
        seeds=table.seed[:count].tolist(), sigmas=table.sigma[:count].tolist())
    return manifest, items


def ensure_manifest(folder, manifest):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Refusing to mix incompatible settings in the same pilot directory')
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
    return digest(manifest)


def read_candidate(path, manifest, index):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        row = json.load(f)
    if (row['manifest_sha256'] != digest(manifest) or row['candidate_index'] != index
            or row['seed'] != manifest['seeds'][index] or row['sigma'] != manifest['sigmas'][index]):
        raise ValueError('Candidate checkpoint does not match frozen manifest')
    expected = sum(manifest[k] for k in ['pool','public','test'])
    if len(row['records']) != expected:
        raise ValueError('Incomplete candidate checkpoint')
    for j,r in enumerate(row['records']):
        if (r['item_index'] != j or not 0 < r['length'] <= manifest['max_tokens']
                or r['finish_reason'] not in ['stop','length']
                or not np.isfinite(r['reward']) or not 0 <= r['reward'] <= 1.100001):
            raise ValueError('Invalid response record')
    return row


def save_candidate(path, manifest, index, records):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    row = dict(manifest_sha256=digest(manifest), candidate_index=index,
               seed=manifest['seeds'][index], sigma=manifest['sigmas'][index], records=records)
    temp = path.with_suffix('.tmp.gz')
    with gzip.open(temp, 'wt', encoding='utf-8') as f:
        json.dump(row, f)
    read_candidate(temp, manifest, index)
    # The complete validated response bundle is the checkpoint; interrupted
    # temporary bundles are never treated as finished candidates.
    os.replace(temp, path)
