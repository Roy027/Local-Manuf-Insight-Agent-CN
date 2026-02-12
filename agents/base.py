from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AgentResult:
    name: str
    status: str
    message: str
    payload: Optional[Any] = None
