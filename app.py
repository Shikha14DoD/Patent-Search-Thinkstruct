"""A simple web page for Part 1's search: the same engine as the CLI, in a browser.

    streamlit run app.py

It is a convenience for trying the search, not the Part 3 "Interface at Scale"
option (Part 3 here is efficiency and large data). The model and index load once
and stay in memory, so after the first search each one takes milliseconds.
"""

from __future__ import annotations

import html
import os
import time

import streamlit as st

from src.search import EXCERPT_MARGIN, Filters, PatentSearch

st.set_page_config(page_title="Patent Evidence Explorer", layout="wide")


@st.cache_resource(show_spinner="Loading the index and the model (once)...")
def load_engine(index_dir: str) -> PatentSearch:
    engine = PatentSearch(index_dir)
    engine.embed_query("warm up")   # load and warm the model now, not in the first search
    if engine.has_hnsw:
        _ = engine.hnsw
    return engine


def show_excerpt(lead: str, label: str, note: str, score: float | None, text: str) -> None:
    details = f"{score:.3f}" if score is not None else ""
    if note:
        details = f"{details}; {note}" if details else note
    heading = f"{lead} {label}" if lead else label
    st.markdown(f"**{html.escape(heading)}**" + (f" ({html.escape(details)})" if details else ""))
    st.markdown(f"> {html.escape(text)}")


with st.sidebar:
    st.header("Filters")
    index_dir = st.text_input("Index folder", "index",
                              help="'index' for Part 1, 'index_large' for Part 3")
    classification = st.text_input("Classification prefix", help="Example: B60B, B60C11")
    title_keywords = st.text_input("Title keywords", help="Example: hub")
    abstract_keywords = st.text_input("Abstract keywords", help='Example: "tread groove" (quotes = exact phrase)')
    either_keywords = st.text_input("Title or abstract keywords", help="Example: sensor")
    exact_title = st.text_input("Exact title", help="Example: PNEUMATIC TIRE")
    match_any = st.checkbox("Keywords: match any word (default: every word)")
    top_k = st.slider("Results", 1, 30, 10)

st.title("Patent Evidence Explorer")
st.caption("Semantic patent search that shows the claim or paragraph behind every result.")

if not os.path.isdir(index_dir):
    st.error(f"No index folder '{index_dir}'. Build it first: python -m src.build_index")
    st.stop()
engine = load_engine(index_dir)

mode = st.radio("Search by", ["Description of an idea or claim", "Similar to a patent ID"],
                horizontal=True)
if mode == "Similar to a patent ID":
    query, similar_to = "", st.text_input("Patent ID", help="Example: US20240051333A1")
else:
    query = st.text_area("Describe the idea, or paste a claim", height=90,
                         help="Example: tire with a sensor that detects loss of pressure")
    similar_to = ""

filters = Filters(classification=classification, title_keywords=title_keywords,
                  abstract_keywords=abstract_keywords,
                  title_or_abstract_keywords=either_keywords,
                  exact_title=exact_title, match_any=match_any)

if not st.button("Search", type="primary"):
    st.info(f"{len(engine.patents):,} patents and {len(engine.units):,} passages in this index.")
    st.stop()

start = time.perf_counter()
source = None
if similar_to.strip():
    source, close = engine.resolve_patent_id(similar_to)
    if source is None:
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        st.error(f"Unknown patent ID '{similar_to}': it is not in this index.{hint}")
        st.stop()
    try:
        report, used = engine.search_similar(source, filters, top_k=top_k)
    except ValueError as error:
        st.error(f"Cannot search for similar patents: {error}.")
        st.stop()
    st.subheader(f"Patents similar to {source}: {engine.patents[source]['title']}")
    st.caption(f"Compared using its {' and '.join(used)}.")
else:
    report = engine.search(query, filters, top_k=top_k)
elapsed_ms = (time.perf_counter() - start) * 1000
ranked = bool(query.strip()) or source is not None

counts = {name: count for name, count in report.filter_counts.items() if name != "combined"}
if counts:
    combined = report.filter_counts.get("combined")
    st.markdown("**Active filters:** " + " · ".join(
        f"{html.escape(name)} ({count} patents)" for name, count in counts.items())
        + (f" → **{combined} patents pass all of them**" if combined is not None else ""))

if not report.results:
    if report.excluded_only:
        st.warning("No results: the filters match only the source patent itself.")
    elif report.empty_reason:
        st.warning(f"No results: the {report.empty_reason} filter matched nothing.")
    else:
        st.warning("No results: each filter matches some patents, but none matches them all.")
    suggestions = [s for s in report.suggestions
                   if source is None or s.kind != "search_by_meaning"]
    if suggestions:
        st.markdown("**Try instead:**")
        for suggestion in suggestions:
            st.markdown(f"- {html.escape(suggestion.message)}")
    st.stop()

if ranked:
    st.caption(f"{report.candidate_patents} of {len(engine.patents)} patents eligible · "
               f"method: {report.strategy} · {elapsed_ms:.0f} ms")
else:
    st.info(f"No query given, so there is nothing to score against: showing the "
            f"{report.candidate_patents} patents that pass the filters, newest first, with "
            f"their abstracts. Add a query to rank them and see matching passages.")
    st.caption(f"{elapsed_ms:.0f} ms")

for result in report.results:
    with st.container(border=True):
        score = f" · score {result.score:.3f}" if ranked else ""
        st.markdown(f"**#{result.rank} [{html.escape(result.title)}]({result.url})**")
        st.caption(f"{result.patent_id} · {result.classification or 'no classification'} · "
                   f"published {result.publication_date}{score}")
        if source is not None and \
                engine.patents[result.patent_id]["abstract"] == engine.patents[source]["abstract"]:
            st.caption("Same abstract as the source patent (possibly the same patent family)")
        if not result.excerpts:
            show_excerpt("", result.excerpt_label, result.excerpt_note, None, result.excerpt_text)
        for number, excerpt in enumerate(result.excerpts):
            show_excerpt("Matched" if number == 0 else "Also matched",
                         excerpt.label, excerpt.note, excerpt.score, excerpt.text)
        if result.more_close_matches:
            plural = "s" if result.more_close_matches > 1 else ""
            st.caption(f"+ {result.more_close_matches} more passage{plural} within "
                       f"{EXCERPT_MARGIN:.2f} of the best match")
