"""Prepare the cross-encoder for the repository: download, quantise to int8, shard.

    python scripts/prepare_model.py                       # msmarco-MiniLM-L12-en-de-v1
    python scripts/prepare_model.py --model cross-encoder/msmarco-MiniLM-L6-en-de-v1

Writes ``model/``: ``model.onnx`` (graph only), ``weights-NN.bin`` (the weights,
each file under ``--max-file-mib``), ``tokenizer.json``, ``config.json`` and
``manifest.json`` (source, revision, sizes, checksums, a fp32-vs-int8 check).

Why sharded: GitHub refuses files over 100 MiB, and this model is about
118 MB even in int8 -- 96 MB of it the 250k-token multilingual vocabulary.
ONNX can keep weights in external files; here every tensor of 1 MiB or more
goes into one of a few shards, so no file comes near the limit and no Git LFS
is needed. Shard names are plain ASCII on purpose: ONNX's own per-tensor
files are named after tensors ("onnx::MatMul_123"), which a Windows checkout
cannot create.

Why int8 dynamic quantisation and not the hub's ``model_qint8_avx512_vnni``:
that file is tuned for one CPU instruction set; whether Vercel's CPUs have it
is not ours to know. ``quantize_dynamic`` with QInt8 weights runs on any x86 CPU.

Needs (build time only, not on Vercel): onnx, onnxruntime, tokenizers,
huggingface_hub, numpy -- and optimum + torch only if the model repository has
no ONNX file to start from.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL = "cross-encoder/msmarco-MiniLM-L12-en-de-v1"
ROOT = Path(__file__).resolve().parent.parent
MiB = 1024 * 1024
SAMPLE = [("Optik: Licht, Linsen, Spiegel", "Sammellinse und Zerstreuungslinse im Versuch"),
          ("Optik: Licht, Linsen, Spiegel", "Bruchrechnen mit Pizza, Klasse 6"),
          ("Technische Informatik", "Logikgatter und Halbaddierer"),
          ("Technische Informatik", "Die Französische Revolution")]


def fetch_fp32(model: str, revision: str, workdir: Path) -> tuple[Path, Path, str]:
    """The fp32 ONNX file, the tokenizer folder, and the exact revision used."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    sha = HfApi().model_info(model, revision=revision).sha or revision
    files = workdir / "hub"
    for name in ("tokenizer.json", "config.json"):
        hf_hub_download(model, name, revision=sha, local_dir=files)
    try:
        onnx_path = Path(hf_hub_download(model, "onnx/model.onnx", revision=sha, local_dir=files))
    except EntryNotFoundError:
        print("Kein onnx/model.onnx im Modell-Repo -- exportiere mit optimum.")
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
        except ImportError as exc:
            raise SystemExit("Export nötig, aber optimum fehlt: siehe scripts/requirements-model.txt") from exc
        exported = workdir / "export"
        ORTModelForSequenceClassification.from_pretrained(model, revision=sha, export=True) \
            .save_pretrained(exported)
        onnx_path = exported / "model.onnx"
    return onnx_path, files, sha


def quantize(fp32: Path, out: Path) -> Path:
    """Dynamic int8 quantisation: weights int8, activations quantised at run time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8, per_channel=False)
    return out


def shard(model_path: Path, out_dir: Path, max_bytes: int, min_tensor_bytes: int = MiB) -> list[Path]:
    """Save ``model_path`` to ``out_dir/model.onnx`` with its large tensors in shards.

    Tensors are packed in order into ``weights-00.bin``, ``weights-01.bin`` ...,
    a new shard starting when the next tensor would cross ``max_bytes``.

    Raises:
        ValueError: when one tensor alone is larger than ``max_bytes``.
    """
    import onnx
    from onnx import numpy_helper
    from onnx.external_data_helper import set_external_data

    model = onnx.load(str(model_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("weights-*.bin"):
        old.unlink()
    shards: list[Path] = []
    handle = None
    used = 0
    try:
        for tensor in model.graph.initializer:
            data = numpy_helper.to_array(tensor).tobytes()
            if len(data) < min_tensor_bytes:
                continue
            if len(data) > max_bytes:
                raise ValueError(f"Tensor {tensor.name} ist mit {len(data) / MiB:.1f} MiB "
                                 f"größer als eine Datei sein darf ({max_bytes / MiB:.0f} MiB).")
            if handle is None or used + len(data) > max_bytes:
                if handle:
                    handle.close()
                shards.append(out_dir / f"weights-{len(shards):02d}.bin")
                handle = shards[-1].open("wb")
                used = 0
            raw = numpy_helper.from_array(numpy_helper.to_array(tensor), tensor.name)
            tensor.CopyFrom(raw)              # normalise to raw_data, whatever field held it
            set_external_data(tensor, location=shards[-1].name, offset=used, length=len(data))
            tensor.ClearField("raw_data")
            tensor.data_location = onnx.TensorProto.EXTERNAL
            handle.write(data)
            used += len(data)
    finally:
        if handle:
            handle.close()
    onnx.save_model(model, str(out_dir / "model.onnx"))
    return [out_dir / "model.onnx", *shards]


def compare(fp32: Path, model_dir: Path) -> list[dict]:
    """Logits of the sample pairs, fp32 vs. the shipped int8 model -- a sanity check."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    sys.path.insert(0, str(ROOT))
    from themenseiten_check.onnx_reranker import encode_pairs, pad_token_id

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    rows = []
    for path in (fp32, model_dir / "model.onnx"):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        scores = []
        for query, doc in SAMPLE:
            enc = encode_pairs(tok, query, [doc])[0]
            feed = {"input_ids": [enc.ids], "attention_mask": [enc.attention_mask],
                    "token_type_ids": [enc.type_ids]}
            import numpy as np
            out = session.run(None, {k: np.array(v, dtype=np.int64) for k, v in feed.items() if k in names})
            scores.append(float(out[0].reshape(-1)[0]))
        rows.append(scores)
    return [{"query": q, "doc": d, "fp32": round(a, 4), "int8": round(b, 4)}
            for (q, d), a, b in zip(SAMPLE, *rows)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(MiB), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument("--out", type=Path, default=ROOT / "model")
    p.add_argument("--max-file-mib", type=float, default=95.0,
                   help="Obergrenze je Datei (GitHub lehnt über 100 MiB ab)")
    args = p.parse_args(argv)
    limit = int(args.max_file_mib * MiB)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fp32, hub, sha = fetch_fp32(args.model, args.revision, work)
        int8 = quantize(fp32, work / "model-int8.onnx")
        if args.out.exists():
            shutil.rmtree(args.out)
        files = shard(int8, args.out, limit)
        for name in ("tokenizer.json", "config.json"):
            shutil.copy(hub / name, args.out / name)
        files += [args.out / "tokenizer.json", args.out / "config.json"]
        too_big = [f for f in files if f.stat().st_size > limit]
        if too_big:
            raise SystemExit(f"Zu groß für GitHub: {', '.join(f.name for f in too_big)}")
        check = compare(fp32, args.out)

    manifest = {
        "model": args.model, "revision": sha, "quantization": "int8 (onnxruntime quantize_dynamic, QInt8)",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in files},
        "total_mib": round(sum(f.stat().st_size for f in files) / MiB, 1),
        "check_fp32_vs_int8": check,
    }
    # Explicit UTF-8: Windows writes cp1252 by default, which a UTF-8 reader rejects.
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("model", "revision", "total_mib")}, ensure_ascii=False))
    for row in check:
        print(f"  fp32 {row['fp32']:+.3f}  int8 {row['int8']:+.3f}  {row['query']} | {row['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
