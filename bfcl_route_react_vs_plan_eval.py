#!/usr/bin/env python3
import argparse, json, os, random, re, time
from typing import Dict, List, Tuple
import pandas as pd
from huggingface_hub import hf_hub_download
from openai import OpenAI
from tqdm import tqdm

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-6-luna")
client = OpenAI()
CACHE: Dict[str, Tuple[str, dict]] = {}

FILES = {
    "single_data": "BFCL_v3_multiple.json",
    "single_ans": "possible_answer/BFCL_v3_multiple.json",
    "multi_data": "BFCL_v3_parallel_multiple.json",
    "multi_ans": "possible_answer/BFCL_v3_parallel_multiple.json",
}


def meta0():
    return dict(llm_calls=0, llm_cache_hits=0, latency_s=0.0,
                input_tokens=0, output_tokens=0)


def add_meta(a, b):
    out = dict(a)
    for k in out:
        out[k] += b.get(k, 0)
    return out


def call_llm(kind, prompt, model, effort, max_output_tokens, use_cache):
    key = f"{model}\n{effort}\n{kind}\n{prompt}"
    if use_cache and key in CACHE:
        text, _ = CACHE[key]
        m = meta0(); m["llm_cache_hits"] = 1
        print(f"      [LLM:{kind}:CACHE] {text!r}", flush=True)
        return text, m

    t0 = time.perf_counter()
    r = client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    dt = time.perf_counter() - t0
    text = r.output_text.strip()
    u = getattr(r, "usage", None)
    m = dict(
        llm_calls=1,
        llm_cache_hits=0,
        latency_s=dt,
        input_tokens=getattr(u, "input_tokens", 0) if u else 0,
        output_tokens=getattr(u, "output_tokens", 0) if u else 0,
    )
    if use_cache:
        CACHE[key] = (text, m)
    print(f"      [LLM:{kind}] {text!r} (lat={dt:.2f}s)", flush=True)
    return text, m


def load_jsonl(filename):
    p = hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset")
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_map(filename):
    return {x["id"]: x for x in load_jsonl(filename)}


def question_text(row):
    out = []
    def walk(x):
        if isinstance(x, dict):
            if x.get("role") == "user" and "content" in x:
                out.append(str(x["content"]))
            else:
                for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(row.get("question", ""))
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def gt_names(ans):
    names = []
    for call in ans.get("ground_truth", []):
        if isinstance(call, dict): names.extend(call.keys())
    return names


def uniq(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def candidate_names(item):
    return [f["name"] for f in item["functions"]]


def tools_text(item):
    blocks = []
    for i, f in enumerate(item["functions"], 1):
        req = (f.get("parameters") or {}).get("required") or []
        blocks.append(
            f"[{i}] {f['name']}\n"
            f"    {re.sub(r'\\s+', ' ', f.get('description','')).strip()}\n"
            f"    required: {', '.join(req) if req else '(none)'}"
        )
    return "\n".join(blocks)


def build_data(n_single, n_multi, seed):
    sd, sa = load_jsonl(FILES["single_data"]), answer_map(FILES["single_ans"])
    md, ma = load_jsonl(FILES["multi_data"]), answer_map(FILES["multi_ans"])

    singles = []
    for r in sd:
        if r["id"] not in sa: continue
        gold_calls = gt_names(sa[r["id"]]); gold = uniq(gold_calls)
        funcs = r.get("function") or []
        if len(gold) == 1 and len(funcs) >= 2:
            singles.append(dict(
                id=r["id"], group="single_tool", gold_route=0,
                question=question_text(r), functions=funcs,
                gold_tools=gold, gold_tool_calls=gold_calls,
            ))

    multis = []
    for r in md:
        if r["id"] not in ma: continue
        gold_calls = gt_names(ma[r["id"]]); gold = uniq(gold_calls)
        funcs = r.get("function") or []
        cands = {f["name"] for f in funcs}
        if 2 <= len(gold) <= 3 and len(funcs) >= 2 and set(gold) <= cands:
            multis.append(dict(
                id=r["id"], group="multi_tool", gold_route=1,
                question=question_text(r), functions=funcs,
                gold_tools=gold, gold_tool_calls=gold_calls,
            ))

    if len(singles) < n_single:
        raise RuntimeError(f"single candidates={len(singles)} < requested={n_single}")
    if len(multis) < n_multi:
        raise RuntimeError(f"multi candidates={len(multis)} < requested={n_multi}")

    rng = random.Random(seed)
    data = rng.sample(singles, n_single) + rng.sample(multis, n_multi)
    rng.shuffle(data)
    return data


def router_prompt(item):
    return f'''Choose the execution strategy for this tool-use request.

Output EXACTLY one character:
0 = ReAct
1 = Plan then Execute

Choose 0 when the user request needs only ONE distinct tool capability, or when it is better to choose the next action only after an observation.
Choose 1 when the user request clearly needs MULTIPLE DISTINCT tool capabilities and identifying them upfront is useful.

Important:
- The number of AVAILABLE tools is not the criterion.
- Several candidate tools may be shown even when only one should be used.
- Judge how many distinct tool capabilities the USER REQUEST actually needs.
- No explanation. No JSON.

User request:
{item['question']}

Available tools:
{tools_text(item)}

Output only 0 or 1.'''


def react_prompt(item, selected):
    return f'''Select the NEXT DISTINCT tool needed for the user request.

Output exactly one exact tool name from Available tools, or output exactly - if no additional distinct tool is needed.

Rules:
- One tool per step.
- Do not repeat an already selected tool.
- No arguments. No explanation. No JSON.

User request:
{item['question']}

Available tools:
{tools_text(item)}

Already selected tools:
{chr(10).join(selected) if selected else '(none)'}

Output one exact tool name or -.'''


def plan_prompt(item):
    return f'''Identify all DISTINCT tools needed to complete this user request.

Output ONLY exact tool names from Available tools, one tool name per line, maximum 3 lines.
If only one tool is needed, output one line.
No bullets. No numbering. No arguments. No explanation. No JSON.

User request:
{item['question']}

Available tools:
{tools_text(item)}

Output only needed exact tool names, one per line.'''


def parse_route(s):
    s = s.strip()
    if s in {"0", "1"}: return int(s)
    m = re.search(r"(?<!\d)([01])(?!\d)", s)
    return int(m.group(1)) if m else 0


def parse_one_tool(s, cands):
    s = s.strip()
    if s == "-": return None
    if s in cands: return s
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if lines and lines[0] in cands: return lines[0]
    hits = [x for x in cands if x in s]
    return hits[0] if len(hits) == 1 else None


def parse_plan(s, cands, max_tools=3):
    out = []
    for line in s.splitlines():
        z = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not z or z == "-": continue
        tool = z if z in cands else None
        if tool is None:
            hits = [x for x in cands if x in z]
            if len(hits) == 1: tool = hits[0]
        if tool and tool not in out: out.append(tool)
        if len(out) >= max_tools: break
    return out


def do_router(item, model, effort, cache):
    raw, m = call_llm("ROUTER", router_prompt(item), model, effort, 8, cache)
    r = parse_route(raw)
    print(f"      [ROUTE] {r} ({'REACT' if r == 0 else 'PLAN_EXECUTE'})", flush=True)
    return r, m


def execute_react(item, max_steps, model, effort, cache):
    selected, total = [], meta0()
    for step in range(1, max_steps + 1):
        print(f"      [REACT STEP {step}/{max_steps}] selected={selected}", flush=True)
        raw, m = call_llm("REACT_NEXT_TOOL", react_prompt(item, selected), model, effort, 40, cache)
        total = add_meta(total, m)
        tool = parse_one_tool(raw, candidate_names(item))
        if tool is None:
            print("      [REACT] -", flush=True)
            break
        if tool in selected:
            print(f"      [REACT] duplicate={tool}; stop", flush=True)
            break
        selected.append(tool)
        print(f"      [REACT] next_tool={tool}", flush=True)
    return selected, total


def execute_plan(item, max_tools, model, effort, cache):
    raw, total = call_llm("PLAN", plan_prompt(item), model, effort, 100, cache)
    selected = parse_plan(raw, candidate_names(item), max_tools)
    print(f"      [PLAN] {selected}", flush=True)
    for i, t in enumerate(selected, 1):
        print(f"          execute #{i}: {t}", flush=True)
    return selected, total


def score_tools(gold, pred):
    g, p = set(gold), set(pred)
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        tool_exact_match=float(g == p),
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1,
        tool_count_match=float(len(g) == len(p)),
        gold_tool_count=len(g),
        pred_tool_count=len(p),
    )


def eval_method(item, method, max_steps, model, effort, cache):
    total = meta0()
    if method == "always_react":
        route = 0
    elif method == "always_plan":
        route = 1
    elif method == "router":
        route, m = do_router(item, model, effort, cache); total = add_meta(total, m)
    elif method == "static_gold":
        route = item["gold_route"]
        print(f"      [STATIC GOLD ROUTE] {route}", flush=True)
    else:
        raise ValueError(method)

    if route == 0:
        pred, m = execute_react(item, max_steps, model, effort, cache)
    else:
        pred, m = execute_plan(item, max_steps, model, effort, cache)
    total = add_meta(total, m)

    sc = score_tools(item["gold_tools"], pred)
    route_acc = float(route == item["gold_route"])
    print(
        f"      [RESULT] route_acc={route_acc:.0f} exact={sc['tool_exact_match']:.0f} "
        f"P={sc['tool_precision']:.3f} R={sc['tool_recall']:.3f} F1={sc['tool_f1']:.3f}\n"
        f"               pred={pred}\n"
        f"               gold={item['gold_tools']}", flush=True
    )

    return dict(
        id=item["id"], group=item["group"], question=item["question"],
        method=method, gold_route=item["gold_route"], pred_route=route,
        route_accuracy=route_acc, **sc,
        gold_tools=" || ".join(item["gold_tools"]), pred_tools=" || ".join(pred),
        llm_calls=total["llm_calls"], llm_cache_hits=total["llm_cache_hits"],
        latency_s=total["latency_s"], input_tokens=total["input_tokens"],
        output_tokens=total["output_tokens"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-single", type=int, default=50)
    ap.add_argument("--n-multi", type=int, default=50)
    ap.add_argument("--max-react-steps", type=int, default=3)
    ap.add_argument("--methods", default="always_react,always_plan,router,static_gold")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--effort", default="none", choices=["none","low","medium","high","xhigh","max"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="bfcl_route_results.csv")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    allowed = {"always_react","always_plan","router","static_gold"}
    if set(methods) - allowed: raise ValueError(set(methods) - allowed)
    cache = not args.no_cache

    print("="*100)
    print("CONFIG")
    print("="*100)
    print(f"model={args.model} effort={args.effort} cache={cache}")
    print(f"n_single={args.n_single} n_multi={args.n_multi} max_react_steps={args.max_react_steps}")
    print(f"methods={methods}\n")

    data = build_data(args.n_single, args.n_multi, args.seed)
    print(f"Loaded {len(data)} examples: {args.n_single} single-tool + {args.n_multi} multi-tool\n")

    rows = []
    for i, item in enumerate(tqdm(data, desc="examples"), 1):
        print("\n" + "="*100)
        print(f"[{i}/{len(data)}] id={item['id']} group={item['group']} gold_route={item['gold_route']}")
        print(f"Q: {item['question']}")
        print("TOOLS:")
        for t in candidate_names(item): print(f"  - {t}")
        print(f"GOLD: {item['gold_tools']}")
        for method in methods:
            print(f"\n  >>> METHOD: {method}")
            try:
                rows.append(eval_method(item, method, args.max_react_steps,
                                        args.model, args.effort, cache))
            except Exception as e:
                print(f"      [ERROR] {repr(e)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    cols = ["route_accuracy","tool_exact_match","tool_precision","tool_recall","tool_f1",
            "tool_count_match","llm_calls","llm_cache_hits","latency_s","input_tokens","output_tokens"]

    print("\n" + "="*100 + "\nOVERALL\n" + "="*100)
    print(df.groupby("method")[cols].mean().reset_index().to_string(index=False))

    print("\n" + "="*100 + "\nBY GROUP\n" + "="*100)
    print(df.groupby(["group","method"])[cols].mean().reset_index().to_string(index=False))

    r = df[df.method == "router"]
    if len(r):
        print("\n" + "="*100 + "\nROUTER CONFUSION\n" + "="*100)
        print(pd.crosstab(r.gold_route, r.pred_route, rownames=["gold"], colnames=["pred"]).to_string())
        print("\nROUTER MIX")
        mix = r.groupby(["group","pred_route"]).size().rename("n").reset_index()
        mix["ratio"] = mix.groupby("group")["n"].transform(lambda s: s/s.sum())
        print(mix.to_string(index=False))

    print(f"\nSaved: {args.output}")
    print(f"Unique cached prompts: {len(CACHE)}")
    if cache:
        print("NOTE: cache ON -> quality is valid, latency is not fair across methods. Use --no-cache for latency comparison.")

if __name__ == "__main__":
    main()
