"""Core requirement quality analysis and local retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import json
import math
import re
from typing import Any, Iterable


DEFAULT_EXPECTED_ROLES = [
    "Business Owner",
    "Product Manager",
    "End User",
    "QA Engineer",
    "Security",
    "Operations",
]

ROLE_PATTERN = re.compile(
    r"\b(?:as|for|by|from)\s+(?:an?\s+)?([a-z][a-z -]{2,40}?)(?=\s+(?:shall|must|should|can|will|needs to|is able to)\b|,|[.;:]|$)",
    re.IGNORECASE,
)
AMBIGUOUS_TERMS = re.compile(
    r"\b(?:quickly|soon|easy|easily|user-friendly|etc\.?|appropriate|reasonable|adequate|as needed|efficiently|seamless(?:ly)?)\b",
    re.IGNORECASE,
)
ACTION_TERMS = re.compile(r"\b(?:shall|must|should|will|needs? to|can)\b", re.IGNORECASE)
ACCEPTANCE_TERMS = re.compile(
    r"\b(?:given|when|then|acceptance criteria|success criteria|so that)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Requirement:
    """A normalized requirement extracted from an uploaded or pasted document."""

    requirement_id: str
    text: str
    source: str = "requirements"


@dataclass
class Finding:
    """A review finding with evidence and a recommended next action."""

    finding_id: str
    category: str
    severity: str
    title: str
    description: str
    requirement_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class AnalysisResult:
    """Machine-readable output from a requirement quality review."""

    requirements: list[Requirement]
    findings: list[Finding]
    retrieved_context: list[Requirement]
    summary: dict[str, Any]
    llm_review: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [asdict(item) for item in self.requirements],
            "findings": [asdict(item) for item in self.findings],
            "retrieved_context": [asdict(item) for item in self.retrieved_context],
            "summary": self.summary,
            "llm_review": self.llm_review,
        }


class LocalRetriever:
    """Small TF-IDF retriever that works without a vector database or API key."""

    def __init__(self, requirements: Iterable[Requirement]) -> None:
        self.requirements = list(requirements)
        self.documents = [self._tokens(item.text) for item in self.requirements]
        document_frequency = Counter(token for document in self.documents for token in set(document))
        count = max(len(self.documents), 1)
        self.idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(document) for document in self.documents]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _vector(self, tokens: Iterable[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {token: (1 + math.log(count)) * self.idf.get(token, 1.0) for token, count in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left.values())) * math.sqrt(
            sum(value * value for value in right.values())
        )
        if not denominator:
            return 0.0
        return sum(left.get(key, 0.0) * value for key, value in right.items()) / denominator

    def search(self, query: str, limit: int = 5) -> list[Requirement]:
        if not self.requirements:
            return []
        query_vector = self._vector(self._tokens(query))
        ranked = sorted(
            zip(self.requirements, self.vectors),
            key=lambda pair: self._cosine(query_vector, pair[1]),
            reverse=True,
        )
        return [requirement for requirement, _ in ranked[: max(limit, 1)]]


def parse_requirements(text: str, source: str = "requirements") -> list[Requirement]:
    """Extract requirements from plain text, Markdown, or pasted numbered lists."""
    requirements: list[Requirement] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        explicit_id = re.match(r"^([A-Za-z]{1,8}[-_ ]?\d+)\s*[:.)-]\s*(.+)$", line)
        numbered = re.match(r"^(\d+)\s*[.)]\s+(.+)$", line)
        if explicit_id:
            requirement_id, requirement_text = explicit_id.groups()
        elif numbered:
            requirement_id, requirement_text = f"REQ-{numbered.group(1)}", numbered.group(2)
        else:
            requirement_id, requirement_text = f"REQ-{line_number}", line
        requirement_text = re.sub(r"\s+", " ", requirement_text).strip()
        if len(requirement_text) >= 12:
            requirements.append(Requirement(requirement_id.upper(), requirement_text, source))
    return requirements


def _normalized(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _roles_in(text: str) -> list[str]:
    roles = []
    for match in ROLE_PATTERN.findall(text):
        role = re.sub(r"\s+", " ", match).strip(" -")
        if role and role.lower() not in {item.lower() for item in roles}:
            roles.append(role.title())
    return roles


def analyze_requirements(
    text: str,
    expected_roles: Iterable[str] | None = None,
    source: str = "requirements",
    retrieval_limit: int = 5,
) -> AnalysisResult:
    """Run deterministic quality checks and retrieve evidence for the review."""
    requirements = parse_requirements(text, source)
    roles = [role.strip() for role in (expected_roles or []) if role.strip()]
    findings: list[Finding] = []
    finding_number = 1

    def add_finding(**kwargs: Any) -> None:
        nonlocal finding_number
        findings.append(Finding(f"F-{finding_number:03d}", **kwargs))
        finding_number += 1

    normalized: dict[str, list[Requirement]] = {}
    for requirement in requirements:
        normalized.setdefault(_normalized(requirement.text), []).append(requirement)
    for matches in normalized.values():
        if len(matches) > 1:
            add_finding(
                category="Duplicate",
                severity="High",
                title="Duplicate requirement",
                description="The same requirement appears more than once.",
                requirement_ids=[item.requirement_id for item in matches],
                evidence=[item.text for item in matches],
                recommendation="Keep one canonical requirement and remove or cross-reference the duplicate.",
            )

    for index, requirement in enumerate(requirements):
        for other in requirements[index + 1 :]:
            if _normalized(requirement.text) == _normalized(other.text):
                continue
            similarity = SequenceMatcher(None, _normalized(requirement.text), _normalized(other.text)).ratio()
            if similarity >= 0.86:
                add_finding(
                    category="Near duplicate",
                    severity="Medium",
                    title="Highly similar requirements",
                    description=f"These requirements overlap by approximately {similarity:.0%}.",
                    requirement_ids=[requirement.requirement_id, other.requirement_id],
                    evidence=[requirement.text, other.text],
                    recommendation="Compare the intended differences and merge or make the distinction explicit.",
                )

    for requirement in requirements:
        if not _roles_in(requirement.text):
            add_finding(
                category="Missing role",
                severity="High",
                title="No responsible actor identified",
                description="The requirement does not state who performs the action or receives the capability.",
                requirement_ids=[requirement.requirement_id],
                evidence=[requirement.text],
                recommendation="Name the actor explicitly, for example: 'As a support agent, I must...'.",
            )
        if not ACTION_TERMS.search(requirement.text):
            add_finding(
                category="Ambiguous",
                severity="Medium",
                title="Missing testable obligation",
                description="The requirement has no clear shall, must, should, will, or needs-to obligation.",
                requirement_ids=[requirement.requirement_id],
                evidence=[requirement.text],
                recommendation="Rewrite it as an observable behavior with a clear obligation and outcome.",
            )
        vague_terms = sorted(set(match.group(0).lower() for match in AMBIGUOUS_TERMS.finditer(requirement.text)))
        if vague_terms:
            add_finding(
                category="Ambiguous",
                severity="Medium",
                title="Vague or subjective wording",
                description=f"The wording contains terms that cannot be verified consistently: {', '.join(vague_terms)}.",
                requirement_ids=[requirement.requirement_id],
                evidence=[requirement.text],
                recommendation="Replace subjective wording with measurable thresholds, timings, or acceptance examples.",
            )
        if not ACCEPTANCE_TERMS.search(requirement.text):
            add_finding(
                category="Missing acceptance criteria",
                severity="Low",
                title="Acceptance criteria not visible",
                description="No acceptance criteria, scenario, or measurable success condition is attached.",
                requirement_ids=[requirement.requirement_id],
                evidence=[requirement.text],
                recommendation="Add Given/When/Then scenarios or measurable acceptance criteria.",
            )

    mentioned_roles = {
        role.title() for requirement in requirements for role in _roles_in(requirement.text)
    }
    for expected_role in roles:
        if expected_role.lower() not in {role.lower() for role in mentioned_roles}:
            add_finding(
                category="Missing role coverage",
                severity="Medium",
                title=f"Expected role not represented: {expected_role}",
                description="The configured stakeholder or delivery role does not appear in any requirement.",
                evidence=[f"Expected role: {expected_role}"],
                recommendation="Confirm whether this role has no responsibilities or add its requirements and acceptance criteria.",
            )

    query = " ".join(
        [
            "requirements duplicate role ambiguity acceptance criteria",
            *(finding.title for finding in findings),
        ]
    )
    retriever = LocalRetriever(requirements)
    retrieved_context = retriever.search(query, retrieval_limit)
    category_counts = Counter(finding.category for finding in findings)
    severity_counts = Counter(finding.severity for finding in findings)
    summary = {
        "requirement_count": len(requirements),
        "finding_count": len(findings),
        "categories": dict(category_counts),
        "severity": dict(severity_counts),
        "roles_detected": sorted(mentioned_roles),
        "expected_roles": roles,
        "retrieval_count": len(retrieved_context),
    }
    return AnalysisResult(requirements, findings, retrieved_context, summary)


def synthesize_with_openai(result: AnalysisResult, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Generate a grounded narrative review from the locally retrieved evidence."""
    from openai import OpenAI

    context = "\n".join(
        f"[{item.requirement_id}] {item.text}" for item in result.retrieved_context
    )
    findings = json.dumps([asdict(item) for item in result.findings], indent=2)
    prompt = f"""You are a senior business analyst and QA reviewer.
Review the requirement-quality findings below using only the retrieved requirement context.
Do not invent requirements or claim that a role is present when it is not in the context.
Return Markdown with sections: Executive summary, Critical issues, Role coverage, Duplicate and ambiguity risks, and Recommended next steps.
Prioritize actionable corrections and cite requirement IDs in square brackets.

Retrieved requirement context:
{context}

Deterministic findings:
{findings}
"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You produce concise, evidence-grounded QA analysis."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "No narrative review was returned."
