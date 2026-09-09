"""Evaluate a model on the OOD set (real document tables) with a chosen prompt.

Three conditions, all greedy (matching verl's val_kwargs: temperature 0, n 1):

  A  base model     + plan prompt (v2)   -> reward, html, schema, content
  B  base model     + html-only prompt   -> html/teds only (no plan asked for)
  C  finetuned      + plan prompt (v2)   -> reward, html, schema, content

    python reward/eval_ood.py --model <path> --prompt {plan,html} --out <jsonl> [--n N]

The OOD parquet stores image BYTES inline rather than paths, so it is self-contained.
Its stored prompt is v1; --prompt plan substitutes v2, which is what the run trained on.

Everything lives under a __main__ guard: vLLM with tensor_parallel_size>1 spawns worker
processes that re-import this module, and module-level work would re-run in each child
("An attempt has been made to start a new process before the current process has finished
its bootstrapping phase").
"""

import argparse
import io
import json
import sys

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/home/ubuntu/verl")
sys.path.insert(0, "/home/ubuntu/table_exps/stage5")

HTML_ONLY = ("Convert this table image to HTML. Output only the HTML table, "
             "starting with <table> and ending with </table>. No explanation, no markdown "
             "fences, no other text.")


def to_img(cell):
    """OOD stores {bytes, path}; train/val store a plain path string."""
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        return Image.open(cell["path"]).convert("RGB")
    return Image.open(cell).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=["plan", "html"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--data", default="/home/ubuntu/table_exps/ood/ood_2k.parquet")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()

    if a.prompt == "plan":
        from rewrite_prompt import PROMPT as P      # v2, exactly what training used
        prompt_text = P.replace("<image>\n", "")
    else:
        prompt_text = HTML_ONLY

    t = pq.read_table(a.data)
    imgs = t.column("images").to_pylist()
    gts = [r["ground_truth"] for r in t.column("reward_model").to_pylist()]
    eis = t.column("extra_info").to_pylist()
    n = a.n or len(gts)
    print(f"{n} rows | model={a.model} | prompt={a.prompt} | tp={a.tp}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util,
              max_model_len=13312, limit_mm_per_prompt={"image": 1})
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens, n=1)

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    reqs = [{"prompt": chat, "multi_modal_data": {"image": to_img(imgs[i][0])}} for i in range(n)]
    outs = llm.generate(reqs, sp)

    with open(a.out, "w") as fh:
        for i, o in enumerate(outs):
            fh.write(json.dumps({
                "i": i, "output": o.outputs[0].text, "gts": gts[i],
                "plan": (eis[i] or {}).get("plan") or "",
                "source": (eis[i] or {}).get("pattern") or "",
            }) + "\n")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
