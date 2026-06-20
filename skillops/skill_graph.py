from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclasses.dataclass
class Action:
    name: str
    args: List[str] = dataclasses.field(default_factory=list)

    def to_str(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": list(self.args)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        return cls(name=d["name"], args=list(d.get("args", [])))


@dataclasses.dataclass
class SkillContract:
    precondition: Dict[str, Any]
    operation: List[Action]
    artifact: Dict[str, Any]
    validator: List[str] = dataclasses.field(default_factory=list)
    failure_modes: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precondition": self.precondition,
            "operation": [a.to_dict() for a in self.operation],
            "artifact": self.artifact,
            "validator": list(self.validator),
            "failure_modes": list(self.failure_modes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillContract":
        return cls(
            precondition=dict(d.get("precondition", {})),
            operation=[Action.from_dict(x) for x in d.get("operation", [])],
            artifact=dict(d.get("artifact", {})),
            validator=list(d.get("validator", [])),
            failure_modes=list(d.get("failure_modes", [])),
        )


@dataclasses.dataclass
class Skill:
    skill_id: str
    name: str
    domain_type: str
    contract: SkillContract
    is_synthetic: bool = False
    parent_skill_id: Optional[str] = None
    degradation_tag: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def precondition(self) -> Dict[str, Any]:
        return self.contract.precondition

    @property
    def operation(self) -> List[Action]:
        return self.contract.operation

    @property
    def artifact(self) -> Dict[str, Any]:
        return self.contract.artifact

    @property
    def validator(self) -> List[str]:
        return self.contract.validator

    @property
    def failure_modes(self) -> List[str]:
        return self.contract.failure_modes

    def signature(self) -> Tuple:
        pre_items = tuple(sorted((k, _hashable(v)) for k, v in self.precondition.items()))
        art_items = tuple(sorted((k, _hashable(v)) for k, v in self.artifact.items()))
        return (self.domain_type, pre_items, art_items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "domain_type": self.domain_type,
            "contract": self.contract.to_dict(),
            "is_synthetic": self.is_synthetic,
            "parent_skill_id": self.parent_skill_id,
            "degradation_tag": self.degradation_tag,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(
            skill_id=d["skill_id"],
            name=d.get("name", d["skill_id"]),
            domain_type=d.get("domain_type", ""),
            contract=SkillContract.from_dict(d.get("contract", {})),
            is_synthetic=bool(d.get("is_synthetic", False)),
            parent_skill_id=d.get("parent_skill_id"),
            degradation_tag=d.get("degradation_tag"),
            metadata=dict(d.get("metadata", {})),
        )


def _hashable(v: Any):
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


EDGE_TYPES = ("dependency", "compatibility", "redundancy", "alternative", "lineage")


@dataclasses.dataclass
class Edge:
    src: str
    dst: str
    edge_type: str
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "edge_type": self.edge_type, "meta": dict(self.meta)}


class SkillLibrary:
    def __init__(self, skills: Optional[Iterable[Skill]] = None) -> None:
        self.skills: Dict[str, Skill] = {}
        self.edges: List[Edge] = []
        self._by_domain_type: Dict[str, List[str]] = defaultdict(list)
        self._by_signature: Dict[Tuple, List[str]] = defaultdict(list)
        if skills:
            for s in skills:
                self.add_skill(s)

    def add_skill(self, skill: Skill) -> None:
        if skill.skill_id in self.skills:
            return
        self.skills[skill.skill_id] = skill
        self._by_domain_type[skill.domain_type].append(skill.skill_id)
        self._by_signature[skill.signature()].append(skill.skill_id)

    def remove_skill(self, skill_id: str) -> None:
        if skill_id not in self.skills:
            return
        s = self.skills.pop(skill_id)
        try:
            self._by_domain_type[s.domain_type].remove(skill_id)
        except ValueError:
            pass
        try:
            self._by_signature[s.signature()].remove(skill_id)
        except ValueError:
            pass
        self.edges = [e for e in self.edges if e.src != skill_id and e.dst != skill_id]

    def add_edge(self, src: str, dst: str, edge_type: str, **meta: Any) -> None:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type}; expected one of {EDGE_TYPES}")
        if src == dst or src not in self.skills or dst not in self.skills:
            return
        self.edges.append(Edge(src=src, dst=dst, edge_type=edge_type, meta=dict(meta)))

    def by_domain_type(self, domain_type: str) -> List[Skill]:
        return [self.skills[i] for i in self._by_domain_type.get(domain_type, [])]

    def all(self) -> List[Skill]:
        return list(self.skills.values())

    def find_by_signature(self, signature: Tuple) -> List[Skill]:
        return [self.skills[i] for i in self._by_signature.get(signature, [])]

    def edges_of(self, skill_id: str, edge_type: Optional[str] = None) -> List[Edge]:
        return [
            e for e in self.edges
            if (e.src == skill_id or e.dst == skill_id)
            and (edge_type is None or e.edge_type == edge_type)
        ]

    def __len__(self) -> int:
        return len(self.skills)

    def build_edges(self) -> Dict[str, int]:
        self.edges = []
        for sids in self._by_signature.values():
            for i, a in enumerate(sids):
                for b in sids[i + 1:]:
                    self.add_edge(a, b, "redundancy", reason="same_signature")
        for dt, sids in self._by_domain_type.items():
            for i, a in enumerate(sids[:200]):
                for b in sids[i + 1: i + 6]:
                    if self.skills[a].signature() != self.skills[b].signature():
                        self.add_edge(a, b, "alternative", reason="same_domain_type")
        for a in self.skills.values():
            for k, v in a.artifact.items():
                for b in self.skills.values():
                    if a.skill_id == b.skill_id:
                        continue
                    if k in b.precondition and b.precondition[k] == v and v not in (None, ""):
                        self.add_edge(a.skill_id, b.skill_id, "dependency", via_key=k, value=v)
        for e in [e for e in self.edges if e.edge_type == "dependency"]:
            a = self.skills[e.src]
            b = self.skills[e.dst]
            ka = e.meta.get("via_key")
            type_a = a.metadata.get("artifact_types", {}).get(ka)
            type_b = b.metadata.get("precondition_types", {}).get(ka)
            if type_a and type_b and type_a == type_b:
                self.add_edge(e.src, e.dst, "compatibility", via_key=ka, type=type_a)
        for s in self.skills.values():
            if s.is_synthetic and s.parent_skill_id and s.parent_skill_id in self.skills:
                self.add_edge(s.parent_skill_id, s.skill_id, "lineage", tag=s.degradation_tag)
        cnt: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            cnt[e.edge_type] += 1
        return dict(cnt)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.skills.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_jsonable(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "SkillLibrary":
        d = json.loads(Path(path).read_text())
        lib = cls()
        for s in d.get("skills", []):
            lib.add_skill(Skill.from_dict(s))
        for e in d.get("edges", []):
            lib.add_edge(e["src"], e["dst"], e["edge_type"], **e.get("meta", {}))
        return lib

    @classmethod
    def load_directory(cls, path) -> "SkillLibrary":
        lib = cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        for fp in sorted(p.iterdir()):
            if fp.suffix.lower() == ".json":
                d = json.loads(fp.read_text())
                lib.add_skill(Skill.from_dict(d))
            elif fp.suffix.lower() in (".yaml", ".yml"):
                try:
                    import yaml
                except ImportError as exc:
                    raise RuntimeError("install pyyaml to load YAML skills") from exc
                d = yaml.safe_load(fp.read_text())
                lib.add_skill(Skill.from_dict(d))
        lib.build_edges()
        return lib
