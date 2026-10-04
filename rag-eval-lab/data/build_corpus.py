"""Build the RAG evaluation corpus from PubMed abstracts via Entrez.

Fixed design decisions (documented here, not chosen after seeing results):
- 4 fixed topic queries, 150 abstracts requested per query, target ~500-600 docs.
- Sort order: Entrez default (most recent first). retstart=0, so the corpus is
  "recent literature on these 4 topics as of the build date".
- Only records with a non-empty abstract are kept.
- Dedupe by PMID across queries.
- Fixed random seed 20261002 is used for any sampling downstream.

Entrez etiquette: no API key, <= 3 requests/second, tool name + email in params.
"""
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SEED = 20261002
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "corpus.jsonl")

QUERIES = {
    "type_2_diabetes_treatment": "type 2 diabetes treatment",
    "hypertension_management": "hypertension management",
    "asthma_inhaler_therapy": "asthma inhaler therapy",
    "migraine_prevention": "migraine prevention treatment",
}
PER_QUERY = 150
BATCH = 200  # efetch ids per request


def get(url, params):
    params = dict(params)
    params["tool"] = "rag-eval-lab"
    params["email"] = "research@example.com"
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(url + "?" + qs)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def esearch_ids(term, retmax):
    raw = get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"},
    )
    data = json.loads(raw)
    return data["esearchresult"]["idlist"]


def efetch_xml(ids):
    raw = get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "rettype": "abstract",
        },
    )
    return raw


def parse_articles(xml_bytes):
    docs = []
    root = ET.fromstring(xml_bytes)
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        title_el = art.find(".//ArticleTitle")
        abs_els = art.findall(".//AbstractText")
        if pmid_el is None or title_el is None or not abs_els:
            continue
        pmid = pmid_el.text.strip()
        title = "".join(title_el.itertext()).strip()
        abstract = " ".join(
            "".join(a.itertext()).strip() for a in abs_els
        ).strip()
        if len(abstract.split()) < 40:
            continue
        mesh = [
            "".join(m.itertext()).strip()
            for m in art.findall(".//MeshHeading/DescriptorName")
        ][:8]
        docs.append(
            {"pmid": pmid, "title": title, "abstract": abstract, "mesh": mesh}
        )
    return docs


def main():
    seen = set()
    docs = []
    for topic, term in QUERIES.items():
        ids = esearch_ids(term, PER_QUERY)
        print(f"{topic}: {len(ids)} ids from esearch")
        time.sleep(0.4)
        for i in range(0, len(ids), BATCH):
            batch = ids[i : i + BATCH]
            parsed = parse_articles(efetch_xml(batch))
            for d in parsed:
                if d["pmid"] in seen:
                    continue
                seen.add(d["pmid"])
                d["topic_query"] = topic
                docs.append(d)
            time.sleep(0.4)
        print(f"  kept so far: {len(docs)}")
    with open(OUT, "w") as f:
        for d in docs:
            d["doc_id"] = f"pmid_{d['pmid']}"
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(docs)} docs to {OUT}")


if __name__ == "__main__":
    main()
