"""Manipulations AST déterministes — opérations structurelles sans LLM."""
from __future__ import annotations

import ast
import logging
from typing import Union

from src.shared.types import ASTOperation

logger = logging.getLogger(__name__)

_DETERMINISTIC_OPS = frozenset({
    "ADD_PARAM", "REMOVE_PARAM", "ADD_RETURN_TYPE", "RENAME_SYMBOL",
    "ADD_IMPORT", "DELETE_NODE", "ADD_DECORATOR", "ADD_DOCSTRING",
    "UPDATE_CALL_SITES",
})


class LLMRequiredError(Exception):
    """L'opération nécessite un LLM — déléguer à ECLMCore."""


class ASTOperationExecutor:
    """Applique des ASTOperations de façon déterministe via ast.NodeTransformer.

    N'appelle jamais de LLM. Pour les ops génératives (MODIFY_BODY, etc.),
    lève LLMRequiredError afin que ECLMCore délègue à Ollama.
    """

    def is_deterministic(self, op_type: str) -> bool:
        """Retourne True si l'opération peut être appliquée sans LLM."""
        return op_type in _DETERMINISTIC_OPS

    def apply(self, code: str, operation: ASTOperation) -> str:
        """Applique l'opération sur le code source et retourne le résultat.

        Args:
            code: Code Python source valide.
            operation: Opération AST à appliquer.

        Returns:
            Code Python modifié (via ast.unparse).

        Raises:
            LLMRequiredError: Si l'opération n'est pas déterministe.
            SyntaxError: Si le code source est syntaxiquement invalide.
        """
        if not self.is_deterministic(operation.op_type):
            raise LLMRequiredError(
                f"{operation.op_type} nécessite un LLM — déléguer à ECLMCore"
            )
        tree = ast.parse(code)
        transformer = self._get_transformer(operation)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

    def _get_transformer(self, op: ASTOperation) -> ast.NodeTransformer:
        handlers = {
            "ADD_PARAM": _AddParamTransformer,
            "REMOVE_PARAM": _RemoveParamTransformer,
            "ADD_RETURN_TYPE": _AddReturnTypeTransformer,
            "RENAME_SYMBOL": _RenameSymbolTransformer,
            "ADD_IMPORT": _AddImportTransformer,
            "DELETE_NODE": _DeleteNodeTransformer,
            "ADD_DECORATOR": _AddDecoratorTransformer,
            "ADD_DOCSTRING": _AddDocstringTransformer,
            "UPDATE_CALL_SITES": _UpdateCallSitesTransformer,
        }
        return handlers[op.op_type](op)

    def update_call_sites_in_project(
        self, project_dir: "Path", old_name: str, new_name: str
    ) -> list["Path"]:
        """Applique UPDATE_CALL_SITES sur tous les .py du dossier projet.

        Args:
            project_dir: Racine du projet à parcourir.
            old_name: Ancien nom du symbole.
            new_name: Nouveau nom du symbole.

        Returns:
            Liste des fichiers modifiés.
        """
        from pathlib import Path
        modified: list[Path] = []
        op = ASTOperation(
            op_type="UPDATE_CALL_SITES",
            target=old_name,
            params={"new_name": new_name},
        )
        for py_file in project_dir.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                # Pré-vérification rapide — évite de parser si le nom n'est pas présent
                if old_name not in source:
                    continue
                result = self.apply(source, op)
                py_file.write_text(result, encoding="utf-8")
                modified.append(py_file)
                logger.info("UPDATE_CALL_SITES: %s → %s dans %s", old_name, new_name, py_file)
            except (OSError, SyntaxError) as exc:
                logger.debug("Skipping %s: %s", py_file, exc)
        return modified


# ── Transformers ──────────────────────────────────────────────────────────────

_FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


class _AddParamTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._op = op

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self._op.target:
            self._add(node)
        self.generic_visit(node)
        return node

    def _add(self, node: ast.FunctionDef) -> None:
        p = self._op.params
        param_name = str(p.get("param_name", "param"))
        param_type = p.get("param_type")
        default_val = p.get("default_value")
        position = int(p.get("position", -1))

        annotation: ast.expr | None = (
            ast.Name(id=str(param_type), ctx=ast.Load()) if param_type else None
        )
        new_arg = ast.arg(arg=param_name, annotation=annotation)

        if position < 0 or position >= len(node.args.args):
            node.args.args.append(new_arg)
            if default_val is not None:
                node.args.defaults.append(ast.Constant(value=default_val))
        else:
            node.args.args.insert(position, new_arg)
            if default_val is not None:
                defaults_start = len(node.args.args) - 1 - len(node.args.defaults)
                rel = position - defaults_start
                if rel >= 0:
                    node.args.defaults.insert(rel, ast.Constant(value=default_val))


class _RemoveParamTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._op = op

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self._op.target:
            name = str(self._op.params.get("param_name", ""))
            before = len(node.args.args)
            node.args.args = [a for a in node.args.args if a.arg != name]
            removed = before - len(node.args.args)
            if removed and node.args.defaults:
                node.args.defaults = node.args.defaults[:-removed]
        self.generic_visit(node)
        return node


class _AddReturnTypeTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._op = op

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self._op.target:
            type_str = str(self._op.params.get("type", "None"))
            node.returns = ast.Name(id=type_str, ctx=ast.Load())
        self.generic_visit(node)
        return node


class _RenameSymbolTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._old = op.target
        self._new = str(op.params.get("new_name", op.target))

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self._old:
            node.id = self._new
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self._old:
            node.name = self._new
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if node.name == self._old:
            node.name = self._new
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg == self._old:
            node.arg = self._new
        return node


class _AddImportTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._module = str(op.params.get("module", ""))
        self._symbol: str | None = op.params.get("symbol")

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # Déduplication
        for stmt in node.body:
            if self._symbol:
                if (
                    isinstance(stmt, ast.ImportFrom)
                    and stmt.module == self._module
                    and any(a.name == self._symbol for a in stmt.names)
                ):
                    return node
            else:
                if isinstance(stmt, ast.Import) and any(
                    a.name == self._module for a in stmt.names
                ):
                    return node

        new_import: ast.stmt
        if self._symbol:
            new_import = ast.ImportFrom(
                module=self._module,
                names=[ast.alias(name=self._symbol)],
                level=0,
            )
        else:
            new_import = ast.Import(names=[ast.alias(name=self._module)])

        node.body.insert(0, new_import)
        return node


class _DeleteNodeTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._target = op.target

    def _filter(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        return [
            s for s in stmts
            if not (
                isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and s.name == self._target  # type: ignore[union-attr]
            )
        ]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._filter(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = self._filter(node.body)
        return node


class _AddDecoratorTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._target = op.target
        self._decorator = str(op.params.get("decorator", "")).lstrip("@")

    def _add(self, node: ast.FunctionDef | ast.ClassDef) -> None:
        if node.name == self._target:
            new_dec: ast.expr = ast.Name(id=self._decorator, ctx=ast.Load())
            node.decorator_list.insert(0, new_dec)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._add(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self._add(node)
        return node


class _UpdateCallSitesTransformer(ast.NodeTransformer):
    """Renomme toutes les références à un symbole dans un fichier.

    Gère : appels de fonctions, accès d'attributs, imports, annotations de type.
    Ne renomme PAS la définition du symbole lui-même (c'est RENAME_SYMBOL).
    """

    def __init__(self, op: ASTOperation) -> None:
        self._old = op.target
        self._new = str(op.params.get("new_name", op.target))

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self._old:
            return ast.Name(id=self._new, ctx=node.ctx)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        if node.attr == self._old:
            node.attr = self._new
        self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        node.names = [
            ast.alias(name=self._new if a.name == self._old else a.name,
                      asname=a.asname)
            for a in node.names
        ]
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        node.names = [
            ast.alias(name=self._new if a.name == self._old else a.name,
                      asname=a.asname)
            for a in node.names
        ]
        return node


class _AddDocstringTransformer(ast.NodeTransformer):
    def __init__(self, op: ASTOperation) -> None:
        self._target = op.target
        self._doc = str(op.params.get("docstring", ""))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self._target:
            doc_node = ast.Expr(value=ast.Constant(value=self._doc))
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            ):
                node.body[0] = doc_node
            else:
                node.body.insert(0, doc_node)
        return node
