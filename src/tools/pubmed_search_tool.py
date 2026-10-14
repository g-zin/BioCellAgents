"""
Literature Evidence Search Tool - PubMed + Europe PMC
=====================================================
Queries PubMed E-utilities and Europe PMC REST API to retrieve
peer-reviewed papers relevant to the classification profile.
No API key required. Free access.
"""

import json
import time
import re
from typing import Type, Dict, Any, List, Optional

import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# -- API Endpoints ----------------------------------------------------------
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/article"

# -- Query Builder ----------------------------------------------------------
ORGAN_KEYWORDS = {
    "head_and_neck": ["head and neck squamous cell carcinoma", "HNSCC"],
    "breast": ["breast carcinoma", "breast cancer"],
    "lung": ["lung carcinoma", "lung cancer"],
    "colorectal": ["colorectal carcinoma", "colon cancer"],
    "prostate": ["prostate carcinoma", "prostate cancer"],
    "skin": ["skin carcinoma", "melanoma"],
    "liver": ["hepatocellular carcinoma", "liver cancer"],
    "stomach": ["gastric carcinoma", "stomach cancer"],
    "pancreas": ["pancreatic carcinoma", "pancreatic cancer"],
    "kidney": ["renal cell carcinoma", "kidney cancer"],
    "bladder": ["bladder carcinoma", "urinary bladder cancer"],
    "cervix": ["cervical carcinoma", "cervical cancer"],
    "ovary": ["ovarian carcinoma", "ovarian cancer"],
    "esophagus": ["esophageal carcinoma", "esophageal cancer"],
    "thyroid": ["thyroid carcinoma", "thyroid cancer"],
    "brain": ["brain tumor", "glioma"],
    "bone": ["bone tumor", "osteosarcoma"],
    "soft_tissue": ["soft tissue sarcoma", "soft tissue tumor"],
    "lymph_node": ["lymph node", "lymphoma"],
    "uterus": ["uterine carcinoma", "endometrial cancer"],
    "unknown": ["carcinoma", "cancer"],
}

TME_KEYWORDS = {
    "immune_desert": ["immune desert", "immune-excluded", "lymphocyte exclusion"],
    "immune_excluded": ["immune excluded", "T-cell exclusion", "stromal barrier"],
    "immune_infiltrated": ["immune infiltrated", "hot tumor", "T-cell infiltrated"],
    "unknown": ["tumor microenvironment", "TME", "immune landscape"],
}


def build_pubmed_query(organ: str, tme_phenotype: str, extra_terms: List[str] = None) -> str:
    """Build a structured PubMed search query from classification results."""
    organ_terms = ORGAN_KEYWORDS.get(organ.lower(), [organ.replace("_", " ")])
    tme_terms = TME_KEYWORDS.get(tme_phenotype.lower(), ["tumor microenvironment"])
    core = f"({organ_terms[0]}) AND ({tme_terms[0]})"
    core += " AND (spatial topology OR spatial biology OR cell-cell interaction)"
    if extra_terms:
        extra = " OR ".join(extra_terms[:3])
        core += f" AND ({extra})"
    return core


def search_pubmed(query: str, retmax: int = 10) -> List[Dict[str, Any]]:
    """Search PubMed and return structured article metadata."""
    params = {
        "db": "pubmed", "term": query, "retmax": retmax,
        "retmode": "json", "sort": "relevance",
        "mindate": "2015/01/01", "maxdate": "2026/12/31",
    }
    try:
        resp = requests.get(PUBMED_ESEARCH, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        time.sleep(0.4)
        summary_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
        resp2 = requests.get(PUBMED_ESUMMARY, params=summary_params, timeout=15)
        resp2.raise_for_status()
        summary_data = resp2.json().get("result", {})
        results = []
        for pmid in pmids:
            art = summary_data.get(pmid, {})
            if not art:
                continue
            authors = [a.get("name", "") for a in art.get("authors", [])]
            author_str = ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else "")
            results.append({
                "pmid": pmid, "title": art.get("title", "Unknown"),
                "authors": author_str,
                "journal": art.get("fulljournalname", art.get("source", "Unknown")),
                "year": art.get("pubdate", "")[:4] or "N/A",
                "abstract": art.get("abstract", ""), "source": "PubMed",
            })
        return results
    except Exception as e:
        print(f"[WARN] PubMed search failed: {e}")
        return []


def search_europe_pmc(query: str, retmax: int = 10) -> List[Dict[str, Any]]:
    """Search Europe PMC and return articles with full-text where available."""
    params = {
        "query": query, "format": "json", "pageSize": retmax,
        "resultType": "core", "sort": "RELEVANCE",
    }
    try:
        resp = requests.get(EUROPE_PMC_SEARCH, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results_list = data.get("resultList", {}).get("result", [])
        results = []
        for art in results_list:
            pmid = art.get("pmid", "")
            pmcid = art.get("pmcid", "")
            doi = art.get("doi", "")
            full_text_snippet = ""
            if pmcid:
                try:
                    time.sleep(0.3)
                    ft_url = f"{EUROPE_PMC_FULLTEXT}/{pmcid}/fullTextXML"
                    ft_resp = requests.get(ft_url, timeout=10)
                    if ft_resp.status_code == 200:
                        text = re.sub(r"<[^>]+>", " ", ft_resp.text)
                        text = re.sub(r"\s+", " ", text).strip()
                        full_text_snippet = text[:2000]
                except Exception:
                    pass
            results.append({
                "pmid": pmid, "pmcid": pmcid, "doi": doi,
                "title": art.get("title", "Unknown"),
                "authors": art.get("authorString", "Unknown"),
                "journal": art.get("journalTitle", art.get("journal", {}).get("title", "Unknown")),
                "year": str(art.get("pubYear", "N/A")),
                "abstract": art.get("abstractInfo", {}).get("abstractText", ""),
                "full_text_snippet": full_text_snippet,
                "isOpenAccess": art.get("isOpenAccess", "N"),
                "source": "Europe PMC",
            })
        return results
    except Exception as e:
        print(f"[WARN] Europe PMC search failed: {e}")
        return []


def merge_results(pubmed_results: List[Dict], epmc_results: List[Dict]) -> List[Dict[str, Any]]:
    """Merge and deduplicate results from both sources."""
    seen_pmid = set()
    merged = []
    for r in pubmed_results:
        key = r.get("pmid") or r.get("title", "").lower()
        if key not in seen_pmid:
            seen_pmid.add(key)
            merged.append(r)
    for r in epmc_results:
        key = r.get("pmid") or r.get("title", "").lower()
        if key not in seen_pmid:
            seen_pmid.add(key)
            merged.append(r)
        else:
            for m in merged:
                if (m.get("pmid") and m["pmid"] == r.get("pmid")) or m.get("title", "").lower() == r.get("title", "").lower():
                    if r.get("full_text_snippet") and not m.get("full_text_snippet"):
                        m["full_text_snippet"] = r["full_text_snippet"]
                    if r.get("doi") and not m.get("doi"):
                        m["doi"] = r["doi"]
                    break
    return merged


# -- CrewAI Tool --------------------------------------------------------------

class LiteratureSearchInput(BaseModel):
    """Input arguments for the Literature Evidence Search Tool."""
    query: str = Field(
        description=(
            "Search query or classification context. Can be a free-text query "
            "or JSON with organ, tme_phenotype fields. "
            "Examples: 'immune desert head and neck squamous cell carcinoma spatial topology', "
            "'{\"organ\": \"head_and_neck\", \"tme_phenotype\": \"immune_desert\"}'"
        )
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of top relevant papers to retrieve (default 5, max 20)",
    )


class LiteratureSearchTool(BaseTool):
    """Searches PubMed and Europe PMC for peer-reviewed literature evidence."""

    name: str = "Literature Evidence Search"
    description: str = (
        "Searches PubMed and Europe PMC for peer-reviewed papers relevant to the "
        "pathology classification profile. Returns citations with PMID, authors, journal, "
        "year, abstract, and full-text snippets where available. "
        "Use this tool to find supporting literature for TME phenotype classifications, "
        "spatial topology findings, immune infiltration patterns, and translational recommendations. "
        "ALWAYS use this tool before making biological claims in clinical reports."
    )
    args_schema: Type[BaseModel] = LiteratureSearchInput

    def _run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute a literature search across PubMed and Europe PMC."""
        organ = "unknown"
        tme_phenotype = "unknown"
        extra_terms = []

        try:
            parsed = json.loads(query)
            if isinstance(parsed, dict):
                organ = parsed.get("organ", "unknown")
                tme_phenotype = parsed.get("tme_phenotype", "unknown")
                extra_terms = parsed.get("extra_terms", [])
                search_query = build_pubmed_query(organ, tme_phenotype, extra_terms)
            else:
                search_query = query
        except (json.JSONDecodeError, TypeError):
            search_query = query

        print(f"[INFO] Literature search query: {search_query}")

        pubmed_results = search_pubmed(search_query, retmax=top_k)
        epmc_results = search_europe_pmc(search_query, retmax=top_k)
        merged = merge_results(pubmed_results, epmc_results)[:top_k]

        if not merged:
            return {
                "status": "no_results",
                "message": f"No relevant literature found for: '{search_query}'",
                "results": [], "total_found": 0,
                "sources_searched": ["PubMed", "Europe PMC"],
            }

        formatted = []
        for i, paper in enumerate(merged, 1):
            citation = (
                f"{paper.get('authors', 'Unknown')} ({paper.get('year', 'N/A')}). "
                f'"{paper.get("title", "Unknown")}." '
                f"{paper.get('journal', 'Unknown')}. PMID: {paper.get('pmid', 'N/A')}."
            )
            entry = {
                "rank": i, "citation": citation,
                "pmid": paper.get("pmid", "N/A"),
                "title": paper.get("title", "Unknown"),
                "authors": paper.get("authors", "Unknown"),
                "journal": paper.get("journal", "Unknown"),
                "year": paper.get("year", "N/A"),
                "abstract": paper.get("abstract", "")[:1500],
                "source": paper.get("source", "Unknown"),
            }
            if paper.get("full_text_snippet"):
                entry["full_text_snippet"] = paper["full_text_snippet"][:1000]
            if paper.get("doi"):
                entry["doi"] = paper["doi"]
            formatted.append(entry)

        return {
            "status": "success", "query": search_query,
            "results": formatted, "total_found": len(formatted),
            "sources_searched": ["PubMed", "Europe PMC"],
        }


if __name__ == "__main__":
    tool = LiteratureSearchTool()
    result = tool._run(
        query="immune desert head and neck squamous cell carcinoma spatial topology",
        top_k=3
    )
    print(json.dumps(result, indent=2))