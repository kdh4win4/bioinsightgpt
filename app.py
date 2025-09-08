# app.py — BioInsightGPT (PubMed Summarizer + Biomedical Entity Graph + Proposal Draft)
# v4: Wide graph, physics OFF, PubTator annotations API, thresholds filter, Research Proposal generator

import io
import math
import textwrap
from datetime import datetime
from itertools import product

import pandas as pd
import requests
import streamlit as st
import networkx as nx
from Bio import Entrez
from pyvis.network import Network
from transformers import pipeline

# ================== STREAMLIT BASIC CONFIG ==================
APP_TITLE = "BioInsightGPT — PubMed Summarizer + Biomedical Entity Graph"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Search ▸ Summarize ▸ Extract Biomedical Entities (Disease / Gene-Protein / Immune Function) "
    "▸ Build Knowledge Graph ▸ Generate Insight ▸ Draft Proposal"
)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Search Settings")
    query = st.text_input("PubMed Query", value="HIV interferon response")
    retmax = st.slider("Number of papers", 10, 200, 50, step=10)
    start_year = st.number_input("Start year", 1990, datetime.now().year, 2015)
    end_year = st.number_input("End year", 1990, datetime.now().year, datetime.now().year)
    Entrez.email = st.text_input("Entrez email (required by NCBI)", value="you@example.com")

    st.divider()
    st.header("NLP Settings")
    summary_len = st.selectbox("Summary length", ["short", "long"], index=0)

    st.divider()
    st.header("Entity Types (Graph)")
    entity_types = st.multiselect(
        "Include entities",
        ["Disease", "GeneProtein", "ImmuneFunction"],
        default=["Disease", "GeneProtein", "ImmuneFunction"],
    )

    st.divider()
    st.header("Graph settings")
    graph_height = st.slider("Graph height (px)", 500, 2000, 900, step=50)
    min_node_freq = st.slider("Min node frequency", 1, 10, 2)
    min_edge_weight = st.slider("Min co-occur weight", 1, 10, 2)

    st.divider()
    st.header("Export")
    export_pdf = st.checkbox("Build a 1-click PDF report", value=True)

# ================== HELPERS ==================
def pubmed_search_and_fetch(query: str, retmax: int, start_year: int, end_year: int):
    """Search PubMed and fetch title/abstract/year/journal/pmid."""
    if not Entrez.email or "@" not in Entrez.email:
        st.error("Please enter a valid Entrez email in the sidebar.")
        st.stop()

    search_term = f'({query}) AND ("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'
    with Entrez.esearch(db="pubmed", term=search_term, retmax=retmax) as handle:
        rec = Entrez.read(handle)
    ids = rec.get("IdList", [])
    if not ids:
        return []

    papers = []
    BATCH = 50
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i+BATCH]
        with Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml") as handle:
            xml = Entrez.read(handle)
        for art in xml.get("PubmedArticle", []):
            try:
                pmid = str(art["MedlineCitation"]["PMID"])
                art_info = art["MedlineCitation"]["Article"]
                title = art_info.get("ArticleTitle", "")
                abs_list = art_info.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join([str(x) for x in abs_list])
                # year
                year = ""
                date_completed = art["MedlineCitation"].get("DateCompleted")
                date_revised = art["MedlineCitation"].get("DateRevised")
                if date_completed and "Year" in date_completed:
                    year = date_completed["Year"]
                elif date_revised and "Year" in date_revised:
                    year = date_revised["Year"]
                else:
                    year = str(start_year)
                journal = art_info.get("Journal", {}).get("Title", "")
                if title or abstract:
                    papers.append(
                        {"pmid": pmid, "title": title, "abstract": abstract, "year": year, "journal": journal}
                    )
            except Exception:
                continue

    return papers

@st.cache_resource(show_spinner=False)
def get_summarizer():
    # 빠른 모델 (품질 높이려면 'facebook/bart-large-cnn')
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(text: str, mode: str = "short"):
    if not text or not text.strip():
        return ""
    summarizer = get_summarizer()
    max_len = 80 if mode == "short" else 200
    min_len = 20 if mode == "short" else 80
    try:
        out = summarizer(text[:4000], max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
        return out[0]["summary_text"]
    except Exception:
        return textwrap.shorten(text, width=200 if mode == "short" else 400, placeholder="…")

# -------- PubTator + Immune term helpers --------
IMMUNE_TERMS = [
    "interferon", "type i interferon", "ifn", "ifn-alpha", "ifn-beta", "ifn-gamma", "ifng",
    "interferon-stimulated gene", "isg", "isgs", "stat1", "jak-stat",
    "tlr", "toll-like receptor", "tlr3", "sting", "cgas",
    "innate immunity", "adaptive immunity", "antiviral response",
    "antigen presentation", "mhc", "hla",
    "nk cell", "natural killer", "t cell", "cd8", "cd4", "b cell",
    "chemokine", "cytokine", "il-6", "il-1", "il-2", "il-15", "il-21", "tnf",
    "apoptosis", "autophagy", "inflammation"
]

PUBTATOR_ANN_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/annotations/"

def _norm_gene(text: str) -> str:
    t = (text or "").strip()
    if len(t) <= 6 and t.isalpha():
        return t.upper()
    return t

def fetch_pubtator_entities(pmids):
    """
    Return {pmid: {'Disease': list, 'GeneProtein': list}} using PubTator annotations API.
    Uses concepts=gene,disease for higher precision.
    """
    out = {str(p): {"Disease": set(), "GeneProtein": set()} for p in pmids}
    if not pmids:
        return {k: {"Disease": [], "GeneProtein": []} for k in out}

    B = 100
    for i in range(0, len(pmids), B):
        batch = pmids[i:i+B]
        try:
            r = requests.get(
                PUBTATOR_ANN_URL,
                params={"pmids": ",".join(batch), "concepts": "gene,disease"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            documents = data.get("documents", data) if isinstance(data, dict) else data
            for doc in documents:
                pmid = str(doc.get("pmid") or doc.get("id") or "")
                if not pmid or pmid not in out:
                    continue
                anns = doc.get("annotations", [])
                if not anns and "passages" in doc:
                    for p in doc.get("passages", []):
                        anns.extend(p.get("annotations", []))
                for a in anns:
                    t = (a.get("type") or a.get("infons", {}).get("type", "")).upper()
                    text = a.get("text") or ""
                    if not text:
                        continue
                    if t == "DISEASE":
                        out[pmid]["Disease"].add(text.strip())
                    elif t in ("GENE", "GENE_PRODUCT", "PROTEIN"):
                        out[pmid]["GeneProtein"].add(_norm_gene(text))
        except Exception:
            continue

    for k in out:
        out[k]["Disease"] = sorted(out[k]["Disease"])
        out[k]["GeneProtein"] = sorted(out[k]["GeneProtein"])
    return out

def extract_immune_functions(text):
    """Very light whitelist matcher for immune function/pathway phrases."""
    txt = (text or "").lower()
    found = []
    for w in IMMUNE_TERMS:
        if w in txt:
            found.append(w)
    return sorted(list(dict.fromkeys(found)))

def build_entity_graph(papers, include_types):
    """
    include_types: ["Disease","GeneProtein","ImmuneFunction"]
    Node: selected entities; Edge: co-occurrence within same paper.
    """
    G = nx.Graph()
    from collections import Counter
    freq = Counter()

    for p in papers:
        bag = []
        for t in include_types:
            bag.extend(p.get(t, []))
        for term in bag:
            freq[term] += 1

    for term, f in freq.items():
        G.add_node(term, size=f)

    for p in papers:
        bag = []
        for t in include_types:
            bag.extend(p.get(t, []))
        uniq = sorted(list(dict.fromkeys(bag)))
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                a, b = uniq[i], uniq[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    return G, freq

def filter_graph_by_thresholds(G, min_node_freq=1, min_edge_weight=1):
    # remove weak edges
    remove_edges = [(u, v) for u, v, w in G.edges(data="weight") if (w or 0) < min_edge_weight]
    G.remove_edges_from(remove_edges)
    # remove small nodes
    remove_nodes = [n for n, attrs in G.nodes(data=True) if (attrs.get("size", 0) < min_node_freq)]
    G.remove_nodes_from(remove_nodes)
    # remove isolated
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return G

def render_pyvis_graph(G, graph_height=900):
    """Return HTML string for Streamlit. Physics OFF; one-time stabilization. Wide width (1800px)."""
    net = Network(height=f"{graph_height}px", width="1800px", directed=False, cdn_resources="in_line")
    net.barnes_hut()
    net.toggle_physics(False)

    for n, attrs in G.nodes(data=True):
        size = 15 + math.sqrt(attrs.get("size", 1)) * 4
        net.add_node(n, label=n, title=f"{n} (freq={attrs.get('size',1)})", value=size)
    for u, v, attrs in G.edges(data=True):
        net.add_edge(u, v, value=attrs.get("weight", 1), title=f"co-occur: {attrs.get('weight',1)}")

    return net.generate_html(notebook=False)

def generate_pdf_report(df):
    """Simple PDF: meta + top 10 summaries + entities."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader  # noqa: F401
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def line(y, txt, size=12):
        c.setFont("Helvetica", size)
        for ln in textwrap.wrap(txt, width=95):
            c.drawString(40, y, ln)
            y -= 16
        return y

    y = height - 40
    y = line(y, APP_TITLE, 16) - 10
    y = line(y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y = line(y, f"Papers: {len(df)}")

    for i, row in df.head(10).iterrows():
        y -= 10
        y = line(y, f"[{i+1}] {row.get('title','')}", 12)
        y = line(y, f"Summary: {row.get('summary','')}", 10)
        y = line(y, "Disease: " + ", ".join(row.get("Disease", [])[:8]), 9)
        y = line(y, "Gene/Protein: " + ", ".join(row.get("GeneProtein", [])[:8]), 9)
        y = line(y, "Immune Function: " + ", ".join(row.get("ImmuneFunction", [])[:8]), 9)
        if y < 120:
            c.showPage()
            y = height - 40

    c.save()
    buf.seek(0)
    return buf

# -------- Proposal generation (gap → hypothesis → experiments) --------
def mine_entities(df):
    diseases = []
    genes = []
    funcs = []
    for _, r in df.iterrows():
        diseases += r.get("Disease", [])
        genes += r.get("GeneProtein", [])
        funcs += r.get("ImmuneFunction", [])
    from collections import Counter
    return Counter(diseases), Counter(genes), Counter(funcs)

def suggest_gaps_and_pairs(G, gene_top=8, func_top=8, min_weight_for_strong=3):
    """
    아이디어: 빈도가 높은 gene/func 중에서 그래프 간선이 없거나(weight<min) 약한 조합을 '연구 갭' 후보로 제안
    """
    # 상위 노드 추출
    deg = dict(G.degree())
    # gene/func 구분이 이름 기반이라 완벽하진 않지만, 그래프 빈도 + 이름 힌트를 함께 사용
    genes = [n for n in sorted(G.nodes(), key=lambda x: deg.get(x,0), reverse=True) if n.isupper() or len(n)<=6][:gene_top]
    funcs = [n for n in sorted(G.nodes(), key=lambda x: deg.get(x,0), reverse=True) if n.lower() != n][:func_top]  # heuristic
    # 보정: 함수 후보에 소문자 기반 키워드도 포함
    funcs = list(dict.fromkeys(funcs + [n for n in G.nodes() if n.islower() and len(n)>2]))[:func_top]

    candidates = []
    for g, f in product(genes, funcs):
        if g == f:
            continue
        w = G[g][f]["weight"] if G.has_edge(g, f) else 0
        if w < min_weight_for_strong:
            candidates.append((g, f, w))
    # weight 오름차순(약할수록 우선), 노드 크기(허브) 고려
    def score(item):
        g, f, w = item
        return (w, -(deg.get(g,0)+deg.get(f,0)))
    candidates.sort(key=score)
    return candidates[:5]  # 상위 5개 갭

def draft_proposal_text(df, G, freq, entity_types, gaps):
    # 토픽 요약
    summaries = " ".join((df["summary"].dropna().tolist()))[:6000]
    brief = summarize("Summarize the key HIV immunology themes in 3 sentences: " + summaries, mode="long")

    # 엔티티 톱 목록
    dis_c, gene_c, func_c = mine_entities(df)
    top_dis = ", ".join([f"{k} (n={v})" for k, v in dis_c.most_common(5)]) or "—"
    top_gene = ", ".join([f"{k} (n={v})" for k, v in gene_c.most_common(8)]) or "—"
    top_func = ", ".join([f"{k} (n={v})" for k, v in func_c.most_common(8)]) or "—"

    # 갭 기반 가설 2–3개
    hypo_lines = []
    for i, (g, f, w) in enumerate(gaps[:3], start=1):
        hypo_lines.append(f"- H{i}. **{g} ↔ {f}** interaction is underexplored (co-occur={w}). We hypothesize that perturbing {g} will modulate **{f}** in HIV context.")

    # 실험 설계(템플릿)
    exp = f"""
**Proposed Experiments**
1) *In vitro*: CRISPRi/siRNA knockdown or pharmacologic modulation of top gene (e.g., {gaps[0][0] if gaps else 'STAT1'}) in CD4⁺ T cells / NK / γδ T cells → measure readouts: ISG panel (qPCR/RNAseq), cytokines, cytotoxicity.
2) *Pathway readouts*: JAK-STAT phosphorylation (pSTAT1/3), IFN-α/β stimulation time-course, flow cytometry functional markers (CD69, CD107a), single-cell RNA-seq if available.
3) *Computational*: Reanalyze public RNA-seq (GEO) for **{gaps[0][0] if gaps else 'STAT1'}**-centric modules; perform GSEA/GSVA for interferon/antiviral signatures; integrate with our entity graph to validate literature coherence.
4) *Clinical signals (if cohort)*: Associate **{gaps[0][0] if gaps else 'STAT1'}** expression/activity with viral load, ISG score, and immune subset frequencies.

**Impact**
- Clarifies how specific gene nodes shape **interferon/antiviral** programs in HIV.
- Prioritizes targets for **adjunct immunomodulation** or **latency control** strategies.
"""

    md = f"""# Research Proposal Draft

**Working Title**  
AI-guided discovery of underexplored gene–immune-function interactions in HIV immunology

**Background (Brief)**  
{brief}

**Top Entities Observed**  
- Disease: {top_dis}  
- Genes/Proteins: {top_gene}  
- Immune Functions: {top_func}

**Graph-Derived Research Gaps (candidates)**  
""" + "\n".join([f"- {g} × {f} (co-occur={w})" for g,f,w in gaps]) + """

**Hypotheses**  
""" + "\n".join(hypo_lines if hypo_lines else ["- (Insufficient gaps detected; increase papers or relax thresholds.)"]) + "\n\n" + exp

    return md

# ================== MAIN LAYOUT ==================
# widen graph area MUCH MORE
col_left, col_right = st.columns([1, 5])

with col_left:
    st.subheader("1) Fetch & Summarize")
    if st.button("🚀 Run pipeline", type="primary"):
        with st.spinner("Searching PubMed & fetching abstracts…"):
            papers = pubmed_search_and_fetch(query, retmax, start_year, end_year)

        if not papers:
            st.warning("No papers found. Try broadening your query or year range.")
            st.stop()

        with st.spinner("Summarizing abstracts…"):
            for p in papers:
                full_text = (p["title"] + ". " + p["abstract"]).strip()
                p["summary"] = summarize(full_text, summary_len)

        # === PubTator 엔티티 + Immune functions ===
        with st.spinner("Extracting biomedical entities (Disease / GeneProtein / ImmuneFunction)…"):
            pmids = [p["pmid"] for p in papers if p.get("pmid")]
            ent_map = fetch_pubtator_entities(pmids)
            for p in papers:
                full_text = (p["title"] + ". " + p["abstract"]).strip()
                p["Disease"] = ent_map.get(p["pmid"], {}).get("Disease", [])
                p["GeneProtein"] = ent_map.get(p["pmid"], {}).get("GeneProtein", [])
                p["ImmuneFunction"] = extract_immune_functions(full_text)

        df = pd.DataFrame(papers)
        st.session_state["papers_df"] = df
        st.success(f"Done! {len(df)} papers processed.")

        # non-empty-entity rows first
        df_display = df.copy()
        df_display["nz_entities"] = df_display[["Disease","GeneProtein","ImmuneFunction"]].apply(
            lambda r: sum(len(r[c]) for c in ["Disease","GeneProtein","ImmuneFunction"]), axis=1
        )
        df_display = df_display.sort_values("nz_entities", ascending=False)

        st.dataframe(
            df_display[["title", "year", "journal", "summary", "Disease", "GeneProtein", "ImmuneFunction"]],
            use_container_width=True,
        )

with col_right:
    st.subheader("2) Entity Graph")
    df = st.session_state.get("papers_df")
    proposal_md = st.session_state.get("proposal_md")
    if df is not None and not df.empty:
        with st.spinner("Building entity co-occurrence graph…"):
            G, freq = build_entity_graph(df.to_dict("records"), include_types=entity_types)
            G_view = filter_graph_by_thresholds(G.copy(), min_node_freq=min_node_freq, min_edge_weight=min_edge_weight)
            html_str = render_pyvis_graph(G_view, graph_height=graph_height)

        st.components.v1.html(html_str, height=graph_height + 40, width=1800, scrolling=True)

        # -------- 3) Proposal Draft --------
        st.subheader("3) Proposal Draft")
        colA, colB = st.columns([1,1])
        with colA:
            st.caption("Find underexplored pairs among high-frequency genes and immune functions, then draft a proposal.")
            gap_weight = st.slider("Treat co-occur < this as 'gap'", 1, 5, 3)
            top_gene = st.slider("Top genes to consider", 3, 15, 8)
            top_func = st.slider("Top immune functions to consider", 3, 15, 8)
        with colB:
            pass

        if st.button("🧪 Generate Proposal"):
            with st.spinner("Mining gaps and drafting proposal…"):
                gaps = suggest_gaps_and_pairs(G, gene_top=top_gene, func_top=top_func, min_weight_for_strong=gap_weight)
                proposal_md = draft_proposal_text(df, G, freq, entity_types, gaps)
                st.session_state["proposal_md"] = proposal_md

        if proposal_md:
            st.markdown(proposal_md)
            st.download_button(
                "⬇️ Download proposal (Markdown)",
                data=proposal_md.encode("utf-8"),
                file_name="proposal_draft.md",
                mime="text/markdown",
            )
        else:
            st.caption("Click **Generate Proposal** to draft a hypothesis & experiments from the current graph.")
    else:
        st.caption("Run the pipeline on the left to see the graph here.")

