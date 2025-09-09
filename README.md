# BioInsightGPT

AI-powered literature mining and knowledge graph tool for biomedical research.  
Built with **Streamlit**, **PubMed/NCBI APIs**, and **transformers** summarization models.

---

## 🚀 Features
- PubMed query + automatic fetching of abstracts
- Abstractive summarization of papers (distilBART)
- Entity extraction:
  - **Diseases** and **Genes/Proteins** (via PubTator3 API)
  - **Immune functions** (keyword dictionary)
- Interactive co-occurrence knowledge graph (PyVis + NetworkX)
- Automated **Research Gap Detection** and **Proposal Draft** generation
- Export summaries & entities to PDF / Markdown

---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/kdh4win4/bioinsightgpt.git
cd bioinsightgpt

