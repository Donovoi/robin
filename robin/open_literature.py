import asyncio
import html
import json
import logging
import math
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aviary.core import Message

from .opencode_llm import RobinLLMClient

logger = logging.getLogger(__name__)

JsonFetcher = Callable[[str, Mapping[str, str] | None], Awaitable[Mapping[str, Any]]]
TextFetcher = Callable[[str, Mapping[str, str] | None], Awaitable[str]]

DEFAULT_USER_AGENT = "RobinOpenLiterature/1.0 (https://github.com/Donovoi/robin)"

OPEN_LITERATURE_SYSTEM_PROMPT = """You are Robin's open literature fallback.

Use only the retrieved records provided by the open literature search layer. Do not invent
citations or claim evidence that is not in the records. Prefer clinical guidelines,
systematic reviews, meta-analyses, randomized trials, mechanistic papers, and primary
biomedical studies over weak secondary mentions. When the evidence conflicts or is thin,
say so directly.

Write a concise but rigorous literature answer with these sections:

1. Bottom line
2. Evidence synthesis
3. Conflicting evidence, gaps, and detractor concerns
4. Research-quality next steps
5. References used

Cite records inline as [R1], [R2], etc. Include DOI, PMID, arXiv ID, or URL in the
references when available."""


@dataclass
class LiteratureRecord:
    title: str
    source_names: set[str]
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_count: int | None = None
    publication_type: str | None = None
    is_open_access: bool = False
    open_access_url: str | None = None
    identifiers: dict[str, str] = field(default_factory=dict)

    def merge(self, other: "LiteratureRecord") -> "LiteratureRecord":
        self.source_names.update(other.source_names)
        self.authors = self.authors or other.authors
        self.venue = self.venue or other.venue
        self.abstract = _prefer_longer_text(self.abstract, other.abstract)
        self.doi = self.doi or other.doi
        self.pmid = self.pmid or other.pmid
        self.pmcid = self.pmcid or other.pmcid
        self.arxiv_id = self.arxiv_id or other.arxiv_id
        self.url = self.url or other.url
        self.open_access_url = self.open_access_url or other.open_access_url
        self.publication_type = self.publication_type or other.publication_type
        self.is_open_access = self.is_open_access or other.is_open_access
        self.identifiers.update({k: v for k, v in other.identifiers.items() if v})
        if other.year and (not self.year or other.year > self.year):
            self.year = other.year
        if other.citation_count is not None:
            self.citation_count = max(self.citation_count or 0, other.citation_count)
        return self


class OpenLiteratureSearcher:
    """Parallel open literature discovery across free scholarly APIs."""

    def __init__(
        self,
        *,
        email: str | None = None,
        semantic_scholar_api_key: str | None = None,
        openalex_api_key: str | None = None,
        max_results_per_source: int = 8,
        timeout: float = 20.0,
        json_fetcher: JsonFetcher | None = None,
        text_fetcher: TextFetcher | None = None,
    ) -> None:
        self.email = _clean_optional(email)
        self.semantic_scholar_api_key = _clean_optional(semantic_scholar_api_key)
        self.openalex_api_key = _clean_optional(openalex_api_key)
        self.max_results_per_source = max_results_per_source
        self.timeout = timeout
        self._json_fetcher = json_fetcher
        self._text_fetcher = text_fetcher

    async def search(self, query: str) -> list[LiteratureRecord]:
        search_query = _compact_search_query(query)
        search_tasks = [
            self._safe_search("OpenAlex", self.search_openalex(search_query)),
            self._safe_search(
                "Semantic Scholar", self.search_semantic_scholar(search_query)
            ),
            self._safe_search("PubMed", self.search_pubmed(search_query)),
            self._safe_search("Europe PMC", self.search_europe_pmc(search_query)),
            self._safe_search("Crossref", self.search_crossref(search_query)),
            self._safe_search("arXiv", self.search_arxiv(search_query)),
        ]
        results = await asyncio.gather(*search_tasks)
        records = [record for source_records in results for record in source_records]
        return self.dedupe_and_rank(records, query=search_query)

    async def search_openalex(self, query: str) -> list[LiteratureRecord]:
        params = {
            "search": query,
            "per-page": str(self.max_results_per_source),
            "sort": "relevance_score:desc",
        }
        if self.email:
            params["mailto"] = self.email
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key
        payload = await self._get_json(_url("https://api.openalex.org/works", params))
        records = []
        for item in payload.get("results", []):
            if not isinstance(item, Mapping):
                continue
            title = _clean_title(item.get("title"))
            if not title:
                continue
            open_access = item.get("open_access") if isinstance(item.get("open_access"), Mapping) else {}
            primary_location = (
                item.get("primary_location")
                if isinstance(item.get("primary_location"), Mapping)
                else {}
            )
            source = (
                primary_location.get("source")
                if isinstance(primary_location.get("source"), Mapping)
                else {}
            )
            authorships = item.get("authorships") if isinstance(item.get("authorships"), list) else []
            authors = []
            for authorship in authorships[:8]:
                if not isinstance(authorship, Mapping):
                    continue
                author = authorship.get("author")
                if isinstance(author, Mapping) and author.get("display_name"):
                    authors.append(str(author["display_name"]))

            doi = _normalise_doi(item.get("doi"))
            record = LiteratureRecord(
                title=title,
                source_names={"OpenAlex"},
                year=_safe_int(item.get("publication_year")),
                authors=authors,
                venue=_clean_optional(source.get("display_name")),
                abstract=_openalex_abstract(item.get("abstract_inverted_index")),
                doi=doi,
                url=doi and f"https://doi.org/{doi}" or _clean_optional(item.get("id")),
                citation_count=_safe_int(item.get("cited_by_count")),
                publication_type=_clean_optional(item.get("type_crossref") or item.get("type")),
                is_open_access=bool(open_access.get("is_oa")),
                open_access_url=_clean_optional(open_access.get("oa_url")),
                identifiers={
                    "openalex": str(item.get("id") or ""),
                    "doi": doi or "",
                },
            )
            records.append(record)
        return records

    async def search_semantic_scholar(self, query: str) -> list[LiteratureRecord]:
        params = {
            "query": query,
            "limit": str(self.max_results_per_source),
            "fields": (
                "paperId,title,year,abstract,url,venue,citationCount,externalIds,"
                "authors,isOpenAccess,openAccessPdf,publicationTypes"
            ),
        }
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key
        payload = await self._get_json(
            _url("https://api.semanticscholar.org/graph/v1/paper/search", params),
            headers=headers,
        )
        records = []
        for item in payload.get("data", []):
            if not isinstance(item, Mapping):
                continue
            title = _clean_title(item.get("title"))
            if not title:
                continue
            external_ids = (
                item.get("externalIds") if isinstance(item.get("externalIds"), Mapping) else {}
            )
            open_pdf = (
                item.get("openAccessPdf")
                if isinstance(item.get("openAccessPdf"), Mapping)
                else {}
            )
            records.append(
                LiteratureRecord(
                    title=title,
                    source_names={"Semantic Scholar"},
                    year=_safe_int(item.get("year")),
                    authors=[
                        str(author.get("name"))
                        for author in item.get("authors", [])
                        if isinstance(author, Mapping) and author.get("name")
                    ][:8],
                    venue=_clean_optional(item.get("venue")),
                    abstract=_clean_optional(item.get("abstract")),
                    doi=_normalise_doi(external_ids.get("DOI")),
                    pmid=_clean_optional(external_ids.get("PubMed")),
                    arxiv_id=_clean_optional(external_ids.get("ArXiv")),
                    url=_clean_optional(item.get("url")),
                    citation_count=_safe_int(item.get("citationCount")),
                    publication_type=", ".join(item.get("publicationTypes", []) or [])
                    or None,
                    is_open_access=bool(item.get("isOpenAccess")),
                    open_access_url=_clean_optional(open_pdf.get("url")),
                    identifiers={
                        "semantic_scholar": str(item.get("paperId") or ""),
                        "doi": _normalise_doi(external_ids.get("DOI")) or "",
                        "pmid": str(external_ids.get("PubMed") or ""),
                        "arxiv": str(external_ids.get("ArXiv") or ""),
                    },
                )
            )
        return records

    async def search_pubmed(self, query: str) -> list[LiteratureRecord]:
        common = {"db": "pubmed", "retmode": "json", "tool": "robin"}
        if self.email:
            common["email"] = self.email
        search_payload = await self._get_json(
            _url(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                {
                    **common,
                    "term": query,
                    "retmax": str(self.max_results_per_source),
                    "sort": "relevance",
                },
            )
        )
        id_list = search_payload.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        summary_payload = await self._get_json(
            _url(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                {**common, "id": ",".join(id_list)},
            )
        )
        result = summary_payload.get("result", {})
        records = []
        for pmid in result.get("uids", []):
            item = result.get(pmid, {})
            if not isinstance(item, Mapping):
                continue
            title = _clean_title(item.get("title"))
            if not title:
                continue
            doi = None
            for article_id in item.get("articleids", []):
                if isinstance(article_id, Mapping) and article_id.get("idtype") == "doi":
                    doi = _normalise_doi(article_id.get("value"))
                    break
            records.append(
                LiteratureRecord(
                    title=title,
                    source_names={"PubMed"},
                    year=_year_from_date(item.get("pubdate") or item.get("epubdate")),
                    authors=[
                        str(author.get("name"))
                        for author in item.get("authors", [])
                        if isinstance(author, Mapping) and author.get("name")
                    ][:8],
                    venue=_clean_optional(item.get("fulljournalname") or item.get("source")),
                    doi=doi,
                    pmid=str(pmid),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    publication_type=", ".join(item.get("pubtype", []) or []) or None,
                    identifiers={"pmid": str(pmid), "doi": doi or ""},
                )
            )
        return records

    async def search_europe_pmc(self, query: str) -> list[LiteratureRecord]:
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(self.max_results_per_source),
            "resultType": "core",
        }
        payload = await self._get_json(
            _url("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params)
        )
        items = payload.get("resultList", {}).get("result", [])
        records = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            title = _clean_title(item.get("title"))
            if not title:
                continue
            pmid = _clean_optional(item.get("pmid"))
            pmcid = _clean_optional(item.get("pmcid"))
            doi = _normalise_doi(item.get("doi"))
            url = _europe_pmc_url(item, pmid)
            records.append(
                LiteratureRecord(
                    title=title,
                    source_names={"Europe PMC"},
                    year=_safe_int(item.get("pubYear")),
                    authors=_split_authors(item.get("authorString")),
                    venue=_clean_optional(item.get("journalTitle")),
                    abstract=_clean_optional(item.get("abstractText")),
                    doi=doi,
                    pmid=pmid,
                    pmcid=pmcid,
                    url=url,
                    citation_count=_safe_int(item.get("citedByCount")),
                    publication_type=_clean_optional(item.get("pubType")),
                    is_open_access=str(item.get("isOpenAccess", "")).upper() == "Y",
                    identifiers={"pmid": pmid or "", "pmcid": pmcid or "", "doi": doi or ""},
                )
            )
        return records

    async def search_crossref(self, query: str) -> list[LiteratureRecord]:
        params = {
            "query.bibliographic": query,
            "rows": str(self.max_results_per_source),
            "select": (
                "DOI,title,author,issued,published-print,published-online,container-title,"
                "abstract,URL,is-referenced-by-count,type"
            ),
        }
        if self.email:
            params["mailto"] = self.email
        payload = await self._get_json(_url("https://api.crossref.org/works", params))
        records = []
        for item in payload.get("message", {}).get("items", []):
            if not isinstance(item, Mapping):
                continue
            title = _clean_title(_first(item.get("title")))
            if not title:
                continue
            authors = []
            for author in item.get("author", [])[:8]:
                if not isinstance(author, Mapping):
                    continue
                name = " ".join(
                    part for part in [author.get("given"), author.get("family")] if part
                ).strip()
                if name:
                    authors.append(name)
            doi = _normalise_doi(item.get("DOI"))
            records.append(
                LiteratureRecord(
                    title=title,
                    source_names={"Crossref"},
                    year=_crossref_year(item),
                    authors=authors,
                    venue=_clean_optional(_first(item.get("container-title"))),
                    abstract=_strip_tags(item.get("abstract")),
                    doi=doi,
                    url=_clean_optional(item.get("URL")) or (doi and f"https://doi.org/{doi}"),
                    citation_count=_safe_int(item.get("is-referenced-by-count")),
                    publication_type=_clean_optional(item.get("type")),
                    identifiers={"doi": doi or ""},
                )
            )
        return records

    async def search_arxiv(self, query: str) -> list[LiteratureRecord]:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(self.max_results_per_source),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        text = await self._get_text(_url("https://export.arxiv.org/api/query", params))
        root = ET.fromstring(text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            title = _clean_title(_xml_text(entry.find("atom:title", ns)))
            if not title:
                continue
            entry_id = _xml_text(entry.find("atom:id", ns))
            arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else None
            authors = [
                _xml_text(author.find("atom:name", ns))
                for author in entry.findall("atom:author", ns)
            ]
            records.append(
                LiteratureRecord(
                    title=title,
                    source_names={"arXiv"},
                    year=_year_from_date(_xml_text(entry.find("atom:published", ns))),
                    authors=[author for author in authors if author][:8],
                    venue="arXiv",
                    abstract=_clean_optional(_xml_text(entry.find("atom:summary", ns))),
                    arxiv_id=arxiv_id,
                    url=entry_id,
                    publication_type="preprint",
                    is_open_access=True,
                    open_access_url=entry_id,
                    identifiers={"arxiv": arxiv_id or ""},
                )
            )
        return records

    @staticmethod
    def dedupe_and_rank(
        records: list[LiteratureRecord],
        *,
        max_records: int | None = None,
        current_year: int | None = None,
        query: str | None = None,
    ) -> list[LiteratureRecord]:
        deduped: dict[str, LiteratureRecord] = {}
        for record in records:
            key = _dedupe_key(record)
            if key in deduped:
                deduped[key].merge(record)
            else:
                deduped[key] = record

        year = current_year or datetime.now(UTC).year
        ranked = sorted(
            deduped.values(),
            key=lambda record: _record_score(
                record, current_year=year, query_terms=_query_terms(query)
            ),
            reverse=True,
        )
        return ranked[:max_records] if max_records is not None else ranked

    async def _safe_search(
        self, source_name: str, coro: Awaitable[list[LiteratureRecord]]
    ) -> list[LiteratureRecord]:
        try:
            return await coro
        except Exception as err:
            logger.warning("Open literature source failed: %s: %s", source_name, err)
            logger.debug("Open literature source failure details", exc_info=True)
            return []

    async def _get_json(
        self, url: str, headers: Mapping[str, str] | None = None
    ) -> Mapping[str, Any]:
        if self._json_fetcher:
            return await self._json_fetcher(url, headers)
        text = await self._get_text(url, headers=headers)
        return json.loads(text)

    async def _get_text(
        self, url: str, headers: Mapping[str, str] | None = None
    ) -> str:
        if self._text_fetcher:
            return await self._text_fetcher(url, headers)
        return await asyncio.to_thread(self._blocking_get_text, url, headers)

    def _blocking_get_text(
        self, url: str, headers: Mapping[str, str] | None = None
    ) -> str:
        request_headers = {"User-Agent": DEFAULT_USER_AGENT, **dict(headers or {})}
        request = urllib.request.Request(url, headers=request_headers)
        context = _ssl_context() if url.startswith("https://") else None
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=context
            ) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {err.code} for {url}: {body[:300]}") from err


async def call_open_literature(
    *,
    queries: dict[str, str],
    llm_client: RobinLLMClient,
    email: str | None = None,
    semantic_scholar_api_key: str | None = None,
    openalex_api_key: str | None = None,
    max_results_per_source: int = 8,
    max_evidence_records: int = 15,
    timeout: float = 20.0,
    max_concurrent_queries: int = 4,
    searcher: OpenLiteratureSearcher | None = None,
) -> dict[str, Any]:
    logger.info(
        "Starting open literature fallback for %s queries across open scholarly APIs.",
        len(queries),
    )
    open_searcher = searcher or OpenLiteratureSearcher(
        email=email,
        semantic_scholar_api_key=semantic_scholar_api_key,
        openalex_api_key=openalex_api_key,
        max_results_per_source=max_results_per_source,
        timeout=timeout,
    )
    semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def process_item(hypothesis: str, query: str) -> dict[str, Any]:
        async with semaphore:
            try:
                records = await open_searcher.search(query)
                evidence_records = OpenLiteratureSearcher.dedupe_and_rank(
                    records, max_records=max_evidence_records, query=query
                )
                if not evidence_records:
                    return {
                        "hypothesis": hypothesis,
                        "query": query,
                        "answer": (
                            "Open literature fallback did not retrieve enough records from "
                            "OpenAlex, Semantic Scholar, PubMed, Europe PMC, Crossref, or arXiv. "
                            "Treat this as an evidence gap and run a broader query or add PDFs."
                        ),
                        "sources": "",
                        "context": f"Query: {query}\nNo open records retrieved.",
                        "status": "success",
                        "task_run_id": f"open-literature:{_slug(hypothesis)}",
                    }

                prompt = _synthesis_prompt(query=query, records=evidence_records)
                response = await llm_client.call_single(
                    [
                        Message(role="system", content=OPEN_LITERATURE_SYSTEM_PROMPT),
                        Message(role="user", content=prompt),
                    ]
                )
                sources = format_literature_sources(evidence_records)
                return {
                    "hypothesis": hypothesis,
                    "query": query,
                    "answer": str(response.text),
                    "sources": sources,
                    "context": f"Query: {query}\nAnswer: {response.text}",
                    "status": "success",
                    "task_run_id": f"open-literature:{_slug(hypothesis)}",
                    "trajectory_url": "",
                }
            except Exception as err:
                logger.exception("Open literature fallback failed for query: %s", query)
                return {
                    "hypothesis": hypothesis,
                    "query": query,
                    "error": f"Open literature fallback failed: {err!s}",
                    "status": "OPEN_LITERATURE_ERROR",
                    "task_run_id": f"open-literature:{_slug(hypothesis)}",
                }

    results = await asyncio.gather(
        *(process_item(hypothesis, query) for hypothesis, query in queries.items())
    )
    has_errors = any(result.get("status") != "success" for result in results)
    return {"results": list(results), "count": len(results), "has_errors": has_errors}


def format_literature_sources(records: list[LiteratureRecord]) -> str:
    lines = []
    for idx, record in enumerate(records, start=1):
        identifiers = _identifier_text(record)
        source_names = ", ".join(sorted(record.source_names))
        venue = f" {record.venue}." if record.venue else ""
        year = f" ({record.year})." if record.year else "."
        url = record.open_access_url or record.url
        url_text = f" {url}" if url else ""
        lines.append(
            f"[R{idx}] {record.title}{year}{venue} Sources: {source_names}."
            f" {identifiers}{url_text}".strip()
        )
    return "\n".join(lines)


def _synthesis_prompt(*, query: str, records: list[LiteratureRecord]) -> str:
    evidence = []
    for idx, record in enumerate(records, start=1):
        authors = ", ".join(record.authors[:6])
        identifiers = _identifier_text(record)
        abstract = _truncate(record.abstract or "No abstract available.", 1400)
        evidence.append(
            "\n".join(
                [
                    f"[R{idx}] {record.title}",
                    f"Year: {record.year or 'unknown'}",
                    f"Authors: {authors or 'unknown'}",
                    f"Venue: {record.venue or 'unknown'}",
                    f"Type: {record.publication_type or 'unknown'}",
                    f"Sources: {', '.join(sorted(record.source_names))}",
                    f"Citations: {record.citation_count if record.citation_count is not None else 'unknown'}",
                    f"Open access: {record.is_open_access}",
                    f"Identifiers: {identifiers or 'none'}",
                    f"URL: {record.open_access_url or record.url or 'none'}",
                    f"Abstract: {abstract}",
                ]
            )
        )

    evidence_text = "\n\n".join(evidence)
    return (
        "Original Robin literature task/query:\n"
        f"{query}\n\n"
        "Retrieved evidence records:\n\n"
        f"{evidence_text}\n\n"
        "Synthesize the answer using only these records. Include detractor concerns: "
        "what could be wrong with the conclusion, what evidence is missing, and what "
        "would change the recommendation."
    )


def _record_score(
    record: LiteratureRecord, *, current_year: int, query_terms: set[str] | None = None
) -> float:
    text = f"{record.title} {record.abstract or ''} {record.publication_type or ''}".lower()
    title_text = record.title.lower()
    score = 0.0
    score += 2.0 * len(record.source_names)
    if record.citation_count:
        score += min(5.0, math.log10(record.citation_count + 1) * 2.0)
    if record.year:
        age = max(0, current_year - record.year)
        score += max(0.0, 3.0 - (age * 0.25))
    if record.is_open_access or record.open_access_url:
        score += 1.0
    if any(term in text for term in ["systematic review", "meta-analysis", "guideline"]):
        score += 5.0
    if any(term in text for term in ["randomized", "randomised", "clinical trial"]):
        score += 3.0
    if any(term in text for term in ["retracted", "withdrawn", "expression of concern"]):
        score -= 10.0
    if record.doi:
        score += 0.5
    for term in query_terms or set():
        if term in title_text:
            score += 0.7
        elif term in text:
            score += 0.25
    return score


def _dedupe_key(record: LiteratureRecord) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    if record.pmid:
        return f"pmid:{record.pmid}"
    if record.pmcid:
        return f"pmcid:{record.pmcid.lower()}"
    if record.arxiv_id:
        return f"arxiv:{record.arxiv_id.lower()}"
    return f"title:{_normalise_title(record.title)}"


def _identifier_text(record: LiteratureRecord) -> str:
    parts = []
    if record.doi:
        parts.append(f"DOI: {record.doi}")
    if record.pmid:
        parts.append(f"PMID: {record.pmid}")
    if record.pmcid:
        parts.append(f"PMCID: {record.pmcid}")
    if record.arxiv_id:
        parts.append(f"arXiv: {record.arxiv_id}")
    return "; ".join(parts)


def _url(base: str, params: Mapping[str, str]) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


def _query_terms(query: str | None) -> set[str]:
    if not query:
        return set()
    stopwords = {
        "about",
        "across",
        "clinical",
        "comprehensive",
        "disease",
        "evidence",
        "literature",
        "papers",
        "relevant",
        "research",
        "review",
        "search",
        "studies",
        "systematic",
        "therapeutic",
    }
    terms = set()
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", query.lower()):
        if term not in stopwords:
            terms.add(term)
    return terms


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _compact_search_query(query: str) -> str:
    query = re.sub(r"<[^>]+>", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(
        r"\b(STRICTLY|JSON|format|Proposal|Overview|Expected Effect|Overall Evaluation)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\s+", " ", query).strip()
    return query[:420]


def _normalise_doi(value: Any) -> str | None:
    cleaned = _clean_optional(value)
    if not cleaned:
        return None
    cleaned = re.sub(r"^https?://(dx\.)?doi\.org/", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.removeprefix("doi:")
    cleaned = cleaned.strip().strip(".")
    return cleaned or None


def _normalise_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _clean_title(value: Any) -> str | None:
    cleaned = _clean_optional(value)
    if not cleaned:
        return None
    return html.unescape(re.sub(r"\s+", " ", cleaned)).strip(" .")


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _prefer_longer_text(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    return right if len(right) > len(left) else left


def _openalex_abstract(index: Any) -> str | None:
    if not isinstance(index, Mapping):
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            position_int = _safe_int(position)
            if position_int is not None:
                words.append((position_int, str(word)))
    return " ".join(word for _, word in sorted(words)) or None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _year_from_date(value: Any) -> int | None:
    cleaned = _clean_optional(value)
    if not cleaned:
        return None
    match = re.search(r"(19|20)\d{2}", cleaned)
    return int(match.group(0)) if match else None


def _crossref_year(item: Mapping[str, Any]) -> int | None:
    for field_name in ["published-print", "published-online", "issued"]:
        date_parts = item.get(field_name, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return _safe_int(date_parts[0][0])
    return None


def _split_authors(value: Any) -> list[str]:
    cleaned = _clean_optional(value)
    if not cleaned:
        return []
    return [part.strip() for part in cleaned.split(",") if part.strip()][:8]


def _europe_pmc_url(item: Mapping[str, Any], pmid: str | None) -> str | None:
    full_text_list = item.get("fullTextUrlList")
    if isinstance(full_text_list, Mapping):
        urls = full_text_list.get("fullTextUrl")
        if isinstance(urls, list):
            for url_record in urls:
                if isinstance(url_record, Mapping):
                    url = _clean_optional(url_record.get("url"))
                    if url:
                        return url
    if pmid:
        return f"https://europepmc.org/article/MED/{pmid}"
    return None


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _strip_tags(value: Any) -> str | None:
    cleaned = _clean_optional(value)
    if not cleaned:
        return None
    return html.unescape(re.sub(r"<[^>]+>", " ", cleaned)).strip()


def _xml_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return re.sub(r"\s+", " ", element.text).strip()


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:80] or "query"
