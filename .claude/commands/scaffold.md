# /scaffold

Crée un nouveau composant ou fichier avec la structure correcte du projet.

## Usage
```
/scaffold component <nom>       ← nouveau composant src/<nom>/
/scaffold script <nom>          ← nouveau script scripts/<nom>.py
/scaffold test <module>         ← nouveau fichier de tests
/scaffold primitive <domaine>   ← template pour une nouvelle primitive
```

## Ce que tu dois faire

### /scaffold component <nom>
Créer la structure complète :
```
src/<nom>/
├── __init__.py          ← exports publics du composant
├── model.py             ← classe principale
├── dataset.py           ← si ML component
└── train.py             ← si ML component
```

Chaque fichier doit :
- Avoir les imports corrects (stdlib → third-party → local)
- Avoir les type hints partout
- Avoir des docstrings Google style
- Passer mypy --strict immédiatement (même vide)

Template `__init__.py` :
```python
"""<Nom> component — <description une ligne>."""
from src.<nom>.model import <MainClass>

__all__ = ["<MainClass>"]
```

Template `model.py` :
```python
"""<Description du composant>."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.shared.config import Config
from src.shared.types import <InputType>, <OutputType>

logger = logging.getLogger(__name__)


class <MainClass>:
    """<Description>."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def <main_method>(self, input: <InputType>) -> <OutputType>:
        """<Description>.

        Args:
            input: <description>

        Returns:
            <description>

        Raises:
            ValueError: si <condition>
        """
        raise NotImplementedError
```

### /scaffold test <module>
```python
"""Tests pour src/<module>/."""
import pytest
from hypothesis import given, strategies as st

from src.<module> import <MainClass>
from src.shared.config import Config


@pytest.fixture
def config() -> Config:
    return Config.for_testing()


@pytest.fixture
def component(config: Config) -> <MainClass>:
    return <MainClass>(config)


class Test<MainClass>:
    def test_nominal_case(self, component: <MainClass>) -> None:
        # TODO
        pass

    def test_edge_case_empty(self, component: <MainClass>) -> None:
        # TODO
        pass

    def test_raises_on_invalid_input(self, component: <MainClass>) -> None:
        with pytest.raises(ValueError):
            # TODO
            pass
```

## Règles
- Toujours créer les tests en même temps que le composant
- Vérifier immédiatement avec `/verify` après scaffold
- Ne jamais créer de fichier sans type hints
