"""Downloads the four benchmarks and their authors' graders, pinned, and writes the question file.

    python -m evaluation.sources --bench bench          # once; ~2 MB of downloads

bench/ afterwards holds
    questions.jsonl     one line per question: id, dataset, grader fields, chat messages, seed_index
    manifest.json       counts, source revisions, sha256 of every downloaded file and of questions.jsonl
    src/                the downloaded files, untouched:
        mbpp/MbppPlus.jsonl                 MBPP+ v0.2.0 (EvalPlus release asset)
        svamp/SVAMP.json                    SVAMP (arkilpatel/SVAMP)
        evalplus/evalplus/...               EvalPlus package (sanitize + correctness check)
        bfcl/bfcl_eval/...                  BFCL v4 data, default prompts, AST checker (gorilla)
        ifeval/instruction_following_eval/  IFEval data and Google's verifiers (google-research)

Every source is fetched at a fixed commit or release, and every file's hash is written to the
manifest on first download and checked on every later run, so a benchmark cannot change silently.
Nothing here is redistributed with the repository; the files belong to their authors.

Question order: each benchmark's questions are shuffled once with Random(42) (MBPP+, then the three
BFCL categories, SVAMP, IFEval, one generator), and seed_index is the position in the resulting
file. The seed of every generation derives from it, which is what lets two runs be paired.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import urllib.parse
import urllib.request

from evaluation.protocol import (BFCL_CATEGORIES, BFCL_PROMPT_FORMAT, MBPP_INSTRUCTION, MBPP_NAME,
                                 QUESTION_ORDER_SEED, SVAMP_SUFFIX)

PINS = {
    "evalplus": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",     # github.com/evalplus/evalplus
    "mbpp": "v0.2.0",                                             # github.com/evalplus/mbppplus_release
    "bfcl": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",          # github.com/ShishirPatil/gorilla
    "svamp": "78e727689e1c1bebfc4be39c446898e8e10b0518",         # github.com/arkilpatel/SVAMP
    "ifeval": "4700efb9afa54286b0e04473ba80a13e8461e25f",        # github.com/google-research/google-research
}
# The MBPP+ release is a tag, which can be re-pointed, so the decompressed file is pinned by hash too.
KNOWN_SHA256 = {"mbpp": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
                "svamp": "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"}
GITHUB_RAW = "https://raw.githubusercontent.com"
BFCL_ROOT = f"{GITHUB_RAW}/ShishirPatil/gorilla/{PINS['bfcl']}/berkeley-function-call-leaderboard"
IFEVAL_ROOT = f"{GITHUB_RAW}/google-research/google-research/{PINS['ifeval']}/instruction_following_eval"
# The two monorepos are large, so only the files the graders import are fetched; EvalPlus is small
# and comes whole, as the tarball of its commit.
BFCL_FILES = (["bfcl_eval/__init__.py", "bfcl_eval/constants/__init__.py",
               "bfcl_eval/constants/enums.py", "bfcl_eval/constants/type_mappings.py",
               "bfcl_eval/constants/default_prompts.py", "bfcl_eval/eval_checker/__init__.py",
               "bfcl_eval/eval_checker/ast_eval/__init__.py",
               "bfcl_eval/eval_checker/ast_eval/ast_checker.py",
               "bfcl_eval/eval_checker/ast_eval/type_convertor/__init__.py",
               "bfcl_eval/eval_checker/ast_eval/type_convertor/java_type_converter.py",
               "bfcl_eval/eval_checker/ast_eval/type_convertor/js_type_converter.py"]
              + [f"bfcl_eval/data/BFCL_v4_{c}.json" for c in BFCL_CATEGORIES]
              + [f"bfcl_eval/data/possible_answer/BFCL_v4_{c}.json"
                 for c in BFCL_CATEGORIES if c != "irrelevance"])
IFEVAL_FILES = ["__init__.py", "instructions.py", "instructions_registry.py", "instructions_util.py",
                "evaluation_lib.py", "data/input_data.jsonl"]


def layout(bench):
    bench = Path(bench)
    src = bench / "src"
    return {"questions": bench / "questions.jsonl", "manifest": bench / "manifest.json",
            "src": src, "mbpp": src / "mbpp/MbppPlus.jsonl", "svamp": src / "svamp/SVAMP.json",
            "evalplus": src / "evalplus", "bfcl": src / "bfcl",
            "ifeval": src / "ifeval/instruction_following_eval"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default="bench")
    arguments = parser.parse_args()
    manifest = prepare(arguments.bench)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"{manifest['total']} questions, sha256 {manifest['questions_sha256'][:12]}")


def prepare(bench):
    """Download what is missing, verify everything against the manifest, (re)write the questions."""
    at = layout(bench)
    download(at)
    old = json.loads(at["manifest"].read_text()) if at["manifest"].exists() else {}
    files = {str(p.relative_to(at["src"])): sha256(p)
             for p in sorted(at["src"].rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    for name, digest in KNOWN_SHA256.items():
        if sha256(at[name]) != digest:
            raise SystemExit(f"{at[name]} is not the pinned file (sha256 mismatch)")
    changed = [name for name, digest in old.get("files", {}).items() if files.get(name) != digest]
    if changed:
        raise SystemExit(f"Downloaded sources differ from the manifest: {changed[:5]}")
    questions = build(at)
    at["questions"].write_text("".join(json.dumps(q) + "\n" for q in questions))
    counts = {}
    for q in questions:
        counts[q["dataset"]] = counts.get(q["dataset"], 0) + 1
    manifest = {"pins": PINS, "counts": counts, "total": len(questions),
                "questions_sha256": sha256(at["questions"]), "files": files}
    if old.get("questions_sha256") not in (None, manifest["questions_sha256"]):
        raise SystemExit("questions.jsonl came out different from the one in the manifest")
    at["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def download(at):
    if not at["mbpp"].exists():
        body = get("https://github.com/evalplus/mbppplus_release/releases/download/"
                   f"{PINS['mbpp']}/MbppPlus.jsonl.gz")
        save(at["mbpp"], gzip.decompress(body))
    if not at["svamp"].exists():
        save(at["svamp"], get(f"{GITHUB_RAW}/arkilpatel/SVAMP/{PINS['svamp']}/SVAMP.json"))
    if not (at["evalplus"] / "evalplus").exists():
        tarball = get(f"https://codeload.github.com/evalplus/evalplus/tar.gz/{PINS['evalplus']}")
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if member.isfile() and len(parts) > 2 and parts[1] == "evalplus":
                    save(at["evalplus"].joinpath(*parts[1:]), archive.extractfile(member).read())
    for name in BFCL_FILES:
        if not (at["bfcl"] / name).exists():
            save(at["bfcl"] / name, get(f"{BFCL_ROOT}/{name}"))
    for name in IFEVAL_FILES:
        if not (at["ifeval"] / name).exists():
            # __init__.py is empty upstream too; the verifiers import the package by this name.
            save(at["ifeval"] / name, b"" if name == "__init__.py" else get(f"{IFEVAL_ROOT}/{name}"))
    import nltk     # some IFEval verifiers split sentences with punkt
    for package in ("punkt", "punkt_tab"):
        if not nltk.download(package, quiet=True):
            raise RuntimeError(f"NLTK {package} could not be downloaded; IFEval needs it")


def build(at):
    rng = random.Random(QUESTION_ORDER_SEED)
    questions = []
    for group in (mbpp_questions(at), *(bfcl_questions(at, c) for c in BFCL_CATEGORIES),
                  svamp_questions(at), ifeval_questions(at)):
        group = list(group)
        rng.shuffle(group)
        questions += group
    for index, question in enumerate(questions):
        question["seed_index"] = index
    return questions


def mbpp_questions(at):
    for task in read_jsonl(at["mbpp"]):
        name = task["entry_point"]
        text = f"{MBPP_INSTRUCTION}\n\n{task['prompt']}\n{MBPP_NAME.format(name=name)}"
        yield {"id": str(task["task_id"]), "dataset": "mbpp", "grader": "mbpp",
               "entry_point": name, "messages": [{"role": "user", "content": text}]}


def bfcl_questions(at, category):
    data = at["bfcl"] / "bfcl_eval/data"
    gold = {}
    if category != "irrelevance":
        gold = {r["id"]: r["ground_truth"]
                for r in read_jsonl(data / f"possible_answer/BFCL_v4_{category}.json")}
    templates = bfcl_templates(at)
    for task in read_jsonl(data / f"BFCL_v4_{category}.json"):
        if len(task["question"]) != 1:
            raise ValueError(f"multi-turn BFCL entry in a single-turn category: {task['id']}")
        system = bfcl_system_prompt(templates, task["function"])
        yield {"id": task["id"], "dataset": f"bfcl_{category}", "grader": "bfcl",
               "category": category, "function": task["function"], "reference": gold.get(task["id"]),
               "messages": [{"role": "system", "content": system}] + task["question"][0]}


def bfcl_templates(at):
    """BFCL's own prompt module, loaded from the pinned file."""
    import importlib.util
    path = at["bfcl"] / "bfcl_eval/constants/default_prompts.py"
    spec = importlib.util.spec_from_file_location("bfcl_default_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.DEFAULT_SYSTEM_PROMPT_FORMAT != BFCL_PROMPT_FORMAT:
        raise ValueError("BFCL's default prompt format is not the one this protocol was fixed on")
    return module


def bfcl_system_prompt(templates, functions):
    """The default format: python call list, no tool-call tag, JSON function docs, plaintext, classic."""
    fmt = dict(urllib.parse.parse_qsl(BFCL_PROMPT_FORMAT))
    style = templates.PROMPT_STYLE_TEMPLATES[fmt["style"]]
    call_format = style["tool_call_no_tag"].format(
        output_format=templates.OUTPUT_FORMAT_MAPPING[fmt["ret_fmt"]],
        param_types=templates.PARAM_TYPE_MAPPING[fmt["ret_fmt"]])
    tools = style["available_tools"].format(format=fmt["func_doc_fmt"],
                                            functions=json.dumps(functions, indent=4))
    return templates.PROMPT_TEMPLATE_MAPPING[fmt["prompt_fmt"]].format(
        persona=style["persona"], task=style["task"], tool_call_format=call_format,
        multiturn_behavior=style["multiturn_behavior"], available_tools=tools)


def svamp_questions(at):
    for task in json.loads(at["svamp"].read_text()):
        answer = str(task["Answer"])
        answer = answer[:-2] if answer.endswith(".0") else answer     # stored as floats
        text = f"{task['Body'].strip()} {task['Question'].strip()}{SVAMP_SUFFIX}"
        yield {"id": f"svamp-{task['ID']}", "dataset": "svamp", "grader": "numeric",
               "answer": answer, "messages": [{"role": "user", "content": text}]}


def ifeval_questions(at):
    for task in read_jsonl(at["ifeval"] / "data/input_data.jsonl"):
        yield {"id": str(task["key"]), "dataset": "ifeval", "grader": "ifeval",
               "instruction_id_list": task["instruction_id_list"], "kwargs": task["kwargs"],
               "original_prompt": task["prompt"],
               "messages": [{"role": "user", "content": task["prompt"]}]}


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
