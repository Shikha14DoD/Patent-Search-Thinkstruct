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

from src.search import EXCERPT_MARGIN, Filters, PatentSearch, Suggestion, date_bound, newest_first

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


DESCRIBE, SIMILAR = "Description of an idea or claim", "Similar to a patent ID"
# Session keys of the inputs, so a suggestion can fill them in.
FILTER_KEYS = {"classification": "classification", "title_keywords": "title_keywords",
               "abstract_keywords": "abstract_keywords",
               "title_or_abstract_keywords": "either_keywords",
               "exact_title": "exact_title", "match_any": "match_any",
               "published_from": "published_from", "published_to": "published_to"}


def apply_suggestion(suggestion: Suggestion) -> None:
    """Put a suggestion's query and filters into the inputs, and search again."""
    for attribute, key in FILTER_KEYS.items():
        st.session_state[key] = getattr(suggestion.filters, attribute)
    if suggestion.query and st.session_state.get("mode") != SIMILAR:
        st.session_state["query"] = suggestion.query
    st.session_state["run"] = True


with st.sidebar:
    st.header("Filters")
    index_dir = st.text_input("Index folder", "index",
                              help="'index' for Part 1, 'index_large' for Part 3")
    classification = st.text_input("Classification prefix", key="classification",
                                   help="Example: B60B, B60C11")
    title_keywords = st.text_input("Title keywords", key="title_keywords", help="Example: hub")
    abstract_keywords = st.text_input("Abstract keywords", key="abstract_keywords",
                                      help='Example: "tread groove" (quotes = exact phrase)')
    either_keywords = st.text_input("Title or abstract keywords", key="either_keywords",
                                    help="Example: sensor")
    exact_title = st.text_input("Exact title", key="exact_title", help="Example: PNEUMATIC TIRE")
    match_any = st.checkbox("Keywords: match any word (default: every word)", key="match_any")
    st.caption("Publication date (the data has no filing dates)")
    date_from, date_to = st.columns(2)
    published_from = date_from.text_input("From", key="published_from",
                                          help="YYYY, YYYY-MM or YYYY-MM-DD, e.g. 2024")
    published_to = date_to.text_input("To", key="published_to",
                                      help="YYYY, YYYY-MM or YYYY-MM-DD, e.g. 2024-06")
    sort = st.radio("Order results by", ["Relevance", "Newest first"], key="sort",
                    help="Newest first shows the same best matches, newest at the top")
    top_k = st.number_input("Results to show", min_value=1, max_value=100_000, value=10, step=10,
                            help="Up to every patent that passes the filters")

st.title("Patent Evidence Explorer")
st.caption("Semantic patent search that shows the claim or paragraph behind every result.")

if not os.path.isdir(index_dir):
    st.error(f"No index folder '{index_dir}'. Build it first: python -m src.build_index")
    st.stop()
engine = load_engine(index_dir)
top_k = min(int(top_k), len(engine.patents))

mode = st.radio("Search by", [DESCRIBE, SIMILAR], horizontal=True, key="mode")
if mode == SIMILAR:
    query, similar_to = "", st.text_input("Patent ID", key="patent_id",
                                          help="Example: US20240051333A1")
else:
    query = st.text_area("Describe the idea, or paste a claim", height=90, key="query",
                         help="Example: tire with a sensor that detects loss of pressure")
    similar_to = ""

filters = Filters(classification=classification, title_keywords=title_keywords,
                  abstract_keywords=abstract_keywords,
                  title_or_abstract_keywords=either_keywords,
                  exact_title=exact_title, match_any=match_any,
                  published_from=published_from, published_to=published_to)
for value in (published_from, published_to):
    if value:
        try:
            date_bound(value, end=False)
        except ValueError as error:
            st.error(f"Publication date: {error}.")
            st.stop()

clicked = st.button("Search", type="primary")
if not (clicked or st.session_state.pop("run", False)):
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
if ranked and sort == "Newest first":
    report.results = newest_first(report.results)

def badge(color: str, text: str) -> str:
    """A coloured label (Streamlit badge markdown); brackets would end it early."""
    return f":{color}-badge[{text.replace('[', '(').replace(']', ')')}]"


counts = {name: count for name, count in report.filter_counts.items() if name != "combined"}
if counts:
    combined = report.filter_counts.get("combined")
    with st.container(border=True):
        chips = [badge("blue" if count else "red", f"{name}: {count} patents")
                 for name, count in counts.items()]
        if combined is not None:
            chips.append(badge("green" if combined else "red",
                               f"{combined} patents pass all filters"))
        st.markdown("**Active filters** &nbsp; " + " &nbsp; ".join(chips))

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
        st.markdown("**Try instead** (click to apply and search again):")
        for number, suggestion in enumerate(suggestions):
            st.button(suggestion.message, key=f"suggestion_{number}",
                      on_click=apply_suggestion, args=(suggestion,))
    st.stop()

if ranked:
    st.markdown(" &nbsp; ".join([
        badge("blue", f"{report.candidate_patents} of {len(engine.patents)} patents eligible"),
        badge("violet", f"method: {report.strategy}"),
        badge("orange", f"{elapsed_ms:.0f} ms")]
        + ([badge("gray", f"best {len(report.results)} matches, newest first")]
           if sort == "Newest first" else [])))
else:
    st.info("No query given, so there is nothing to score against: showing the patents that "
            "pass the filters, newest first, with their abstracts. Add a query to rank them "
            "and see matching passages.")
    st.markdown(badge("orange", f"{elapsed_ms:.0f} ms"))

eligible = report.candidate_patents
if eligible > len(report.results):
    st.info(f"Showing {len(report.results)} of {eligible} patents that pass the filters. "
            "Increase 'Results to show' in the sidebar to see more.")

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