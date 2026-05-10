"""Construit le dataset de curriculum pour l'entraînement GRPO de l'ECLMCore.

Sources de données (par ordre de priorité) :
  1. Fichiers Python existants du projet (functions/classes)
  2. Paires DPO déjà collectées (chosen = haute qualité)
  3. Répertoire externe de code Python fourni via --source-dir

Usage:
    python scripts/build_curriculum.py [--source-dir /path/to/python/code]
    python scripts/build_curriculum.py --source-dir /path/to/repos --min-examples 1000
"""
from __future__ import annotations

import ast
import json
import logging
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Ajouter le projet au sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.config import Config

logger = logging.getLogger(__name__)

MAX_FUNCTION_LINES = 80     # Exclure les fonctions trop longues
MIN_FUNCTION_LINES = 3      # Exclure les triviales (pass, return None)
COMPLEXITY_THRESHOLDS = [0, 10, 25, 50, 80]  # lignes → complexity 1..5


@dataclass
class RawExample:
    """Exemple brut extrait d'un fichier Python."""
    source_file: str
    function_name: str
    function_code: str      # corps isolé
    context_code: str       # tout ce qui précède dans le fichier (imports + defs)
    tests: list[str]        # tests pytest extraits du fichier de test associé
    complexity: int         # 1-5


# ── Analyse AST ───────────────────────────────────────────────────────────────

def _count_ast_nodes(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _estimate_complexity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    lines = (func_node.end_lineno or 0) - func_node.lineno
    for i, threshold in enumerate(COMPLEXITY_THRESHOLDS[1:], 1):
        if lines < threshold:
            return i
    return 5


def _function_source(source_lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "\n".join(source_lines[start:end])


def _context_before(source_lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Importe + définitions avant la fonction (classes parentes incluses)."""
    end = node.lineno - 1
    ctx_lines = source_lines[:end]
    # Garder max 60 lignes de contexte
    return "\n".join(ctx_lines[-60:])


def extract_functions(py_file: Path) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[str]]]:
    """Extrait les fonctions/méthodes de niveau module."""
    try:
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return []
    source_lines = source.splitlines()
    results: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue  # skip private helpers
        lines = (node.end_lineno or 0) - node.lineno
        if not MIN_FUNCTION_LINES <= lines <= MAX_FUNCTION_LINES:
            continue
        results.append((node, source_lines))
    return results


def _find_tests_for(py_file: Path, function_name: str) -> list[str]:
    """Cherche les fonctions pytest qui testent `function_name`."""
    tests_dir = py_file.parent.parent / "tests"
    candidates = [
        tests_dir / f"test_{py_file.stem}.py",
        py_file.parent / f"test_{py_file.stem}.py",
    ]
    for test_path in candidates:
        if not test_path.exists():
            continue
        try:
            source = test_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            body_src = ast.unparse(node)
            if function_name in body_src:
                lines = source.splitlines()
                start = node.lineno - 1
                end = node.end_lineno or node.lineno
                found.append("\n".join(lines[start:end]))
        if found:
            return found
    return []


# ── Scoring ────────────────────────────────────────────────────────────────────

def _ruff_ok(code: str) -> bool:
    result = subprocess.run(
        ["ruff", "check", "--select=E9,F", "-"],
        input=code, capture_output=True, text=True,
    )
    return result.returncode == 0


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _score_example(func_code: str, ctx_code: str) -> float:
    full = ctx_code + "\n\n" + func_code
    if not _syntax_ok(full):
        return 0.0
    return 0.9 if _ruff_ok(full) else 0.6


# ── Dataset building ───────────────────────────────────────────────────────────

def collect_from_directory(src_dir: Path) -> list[RawExample]:
    """Parcourt src_dir et extrait les fonctions valides."""
    examples: list[RawExample] = []
    py_files = [p for p in src_dir.rglob("*.py")
                if "test_" not in p.name and "__pycache__" not in str(p)]

    for py_file in py_files:
        for func_node, source_lines in extract_functions(py_file):
            func_code = _function_source(source_lines, func_node)
            ctx_code = _context_before(source_lines, func_node)

            if _score_example(func_code, ctx_code) < 0.5:
                continue

            tests = _find_tests_for(py_file, func_node.name)
            complexity = _estimate_complexity(func_node)
            examples.append(RawExample(
                source_file=str(py_file),
                function_name=func_node.name,
                function_code=func_code,
                context_code=ctx_code,
                tests=tests,
                complexity=complexity,
            ))
    return examples


def _to_curriculum_line(ex: RawExample) -> dict[str, object]:
    """Convertit un RawExample en ligne JSONL pour le curriculum."""
    return {
        "operation": {
            "op_type": "CREATE_FUNCTION",
            "target": ex.function_name,
            "params": {
                "description": f"Implement the function `{ex.function_name}` in Python",
                "target_type": "function",
            },
        },
        # current_code = fichier avec la fonction remplacée par un stub
        "current_code": ex.context_code,
        "target_code": ex.function_code,
        "reward": 1.0,          # code existant = ground truth
        "complexity": ex.complexity,
        "tests": ex.tests,
    }


def collect_from_dpo(dpo_dir: Path) -> list[dict[str, object]]:
    """Importe directement les paires DPO comme exemples CREATE_MODULE."""
    lines: list[dict[str, object]] = []
    for path in sorted(dpo_dir.glob("dpo_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pair = json.loads(line)
                    if float(pair.get("chosen_score", 0)) < 0.8:
                        continue
                    lines.append({
                        "operation": {
                            "op_type": "CREATE_MODULE",
                            "target": str(pair.get("prompt", ""))[:50],
                            "params": {"description": str(pair.get("prompt", ""))},
                        },
                        "current_code": "",
                        "target_code": str(pair["chosen"]),
                        "reward": float(pair.get("chosen_score", 1.0)),
                        "complexity": 2,
                        "tests": [],
                    })
                except (KeyError, ValueError):
                    pass
    return lines


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Construit le dataset de curriculum GRPO")
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Répertoire Python supplémentaire à parcourir")
    parser.add_argument("--min-examples", type=int, default=500,
                        help="Nombre minimum d'exemples requis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(args.seed)

    config = Config()
    out_dir = config.data_dir / "training" / "eclm"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_lines: list[dict[str, object]] = []

    # Source 1 : projet ECLM lui-même
    eclm_src = config.root_dir / "src"
    if eclm_src.exists():
        examples = collect_from_directory(eclm_src)
        logger.info("Projet ECLM: %d exemples extraits", len(examples))
        all_lines.extend(_to_curriculum_line(ex) for ex in examples)

    # Source 2 : répertoire externe
    if args.source_dir and args.source_dir.exists():
        ext_examples = collect_from_directory(args.source_dir)
        logger.info("Source externe: %d exemples extraits", len(ext_examples))
        all_lines.extend(_to_curriculum_line(ex) for ex in ext_examples)

    # Source 3 : paires DPO
    if config.dpo_pairs_dir.exists():
        dpo_lines = collect_from_dpo(config.dpo_pairs_dir)
        logger.info("Paires DPO: %d exemples", len(dpo_lines))
        all_lines.extend(dpo_lines)

    if len(all_lines) < args.min_examples:
        logger.warning(
            "Seulement %d exemples (minimum %d). "
            "Fournissez --source-dir avec plus de code Python.",
            len(all_lines), args.min_examples,
        )

    # Mélanger et répartir par complexité dans des fichiers séparés
    random.shuffle(all_lines)
    by_complexity: dict[int, list[dict[str, object]]] = {i: [] for i in range(1, 6)}
    for line in all_lines:
        c = int(str(line.get("complexity", 1)))
        by_complexity[min(5, max(1, c))].append(line)

    total_written = 0
    for complexity, lines in by_complexity.items():
        if not lines:
            continue
        out_path = out_dir / f"curriculum_{complexity}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        logger.info("curriculum_%d.jsonl — %d exemples", complexity, len(lines))
        total_written += len(lines)

    logger.info("Total: %d exemples dans %s", total_written, out_dir)

    if total_written >= args.min_examples:
        logger.info("Prêt pour l'entraînement: python -m src.eclm.train")
    else:
        logger.warning(
            "Pas assez d'exemples. Fournissez un repo Python externe:\n"
            "  python scripts/build_curriculum.py --source-dir ~/code/my-project"
        )


if __name__ == "__main__":
    main()
