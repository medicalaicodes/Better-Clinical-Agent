"""
The medical knowledge corpora we search.

We use exactly the snippet-level corpora of MedRAG / MIRAGE (Xiong et al., ACL Findings 2024),
so our retrieval is comparable with the published baselines:

  Textbooks   18 USMLE textbooks (from MedQA), ~125.8k snippets.
              Downloaded from the HuggingFace dataset `MedRAG/textbooks`.
  StatPearls  ~9.3k clinical decision-support articles, ~301k snippets.
              Not on HuggingFace, so we download NCBI's dump (~2 GB) and cut it into snippets
              with a port of MedRAG's own chunker, so our snippet boundaries match theirs.

On disk, each corpus is a folder of jsonl files:
    <corpus_dir>/<name>/chunk/*.jsonl      one {"id", "title", "content"} per line

All corpora are then merged into one in-memory CorpusStore. A snippet's row number in the
store is its id inside every index (BM25 and FAISS), so the row order must never change
between building and loading. That is why the store is saved as one consolidated file.
"""

import json
import os
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

STATPEARLS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"


@dataclass(frozen=True)
class Snippet:
    """One retrievable unit of text."""
    id: str          # e.g. "First_Aid_Step2_496" or "article-12345_3"
    title: str       # book name, or "Article -- Section -- Subsection" for StatPearls
    content: str
    source: str      # "textbooks" | "statpearls"

    @property
    def text(self) -> str:
        """Title + content: the form we give to BM25 and the cross-encoder.
        The title carries the topic ("Pulmonary Embolism -- Evaluation") that the
        paragraph itself often never repeats."""
        if self.title:
            return f"{self.title}. {self.content}"
        return self.content


class CorpusStore:
    """All snippets, stored column by column (four parallel lists) to keep memory low."""

    def __init__(self, snippets: Sequence[Snippet]):
        self.ids: List[str] = [s.id for s in snippets]
        self.titles: List[str] = [s.title for s in snippets]
        self.contents: List[str] = [s.content for s in snippets]
        self.sources: List[str] = [s.source for s in snippets]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, row: int) -> Snippet:
        return Snippet(self.ids[row], self.titles[row], self.contents[row], self.sources[row])

    def texts(self) -> Iterator[str]:
        for row in range(len(self)):
            yield self[row].text

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in range(len(self)):
                record = {"id": self.ids[row], "title": self.titles[row],
                          "content": self.contents[row], "source": self.sources[row]}
                f.write(json.dumps(record) + "\n")

    @classmethod
    def load(cls, path: str) -> "CorpusStore":
        snippets = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                snippets.append(Snippet(record["id"], record["title"], record["content"], record["source"]))
        return cls(snippets)


# ----------------------------------------------------------------------------------------
# Reading chunked corpora
# ----------------------------------------------------------------------------------------

def iter_chunk_dir(chunk_dir: str, source: str) -> Iterator[Snippet]:
    """Yields every snippet from a MedRAG-style folder of jsonl files (sorted, so the order
    is the same on every machine)."""
    for file_name in sorted(os.listdir(chunk_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        with open(os.path.join(chunk_dir, file_name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield Snippet(id=record["id"], title=record.get("title", ""),
                              content=record["content"], source=source)


def load_corpora(root: str, names: Iterable[str]) -> CorpusStore:
    snippets: List[Snippet] = []
    for name in names:
        chunk_dir = os.path.join(root, name, "chunk")
        if not os.path.isdir(chunk_dir):
            raise FileNotFoundError(f"Corpus '{name}' not found at {chunk_dir}. "
                                    f"Run: python scripts/build_index.py download --corpora {name}")
        snippets.extend(iter_chunk_dir(chunk_dir, source=name))
    return CorpusStore(snippets)


# ----------------------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------------------

def download_textbooks(root: str) -> str:
    """MedRAG/textbooks from HuggingFace: already chunked, ~210 MB."""
    from huggingface_hub import snapshot_download

    destination = os.path.join(root, "textbooks")
    snapshot_download(repo_id="MedRAG/textbooks", repo_type="dataset",
                      allow_patterns=["chunk/*"], local_dir=destination)
    return destination


def download_statpearls(root: str, keep_archive: bool = False) -> str:
    """Downloads the NCBI StatPearls dump (~2 GB), unpacks it, and chunks every article."""
    import requests
    from tqdm import tqdm

    destination = os.path.join(root, "statpearls")
    os.makedirs(destination, exist_ok=True)
    archive = os.path.join(destination, "statpearls_NBK430685.tar.gz")
    raw_dir = os.path.join(destination, "statpearls_NBK430685")

    # 1. Download (to a .part file first, so an interrupted download is never mistaken
    #    for a finished one) and unpack.
    if not os.path.isdir(raw_dir):
        if not os.path.exists(archive):
            with requests.get(STATPEARLS_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with open(archive + ".part", "wb") as f, \
                        tqdm(total=total, unit="B", unit_scale=True, desc="StatPearls") as bar:
                    for block in response.iter_content(chunk_size=1 << 20):
                        f.write(block)
                        bar.update(len(block))
            os.replace(archive + ".part", archive)

        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter="data")
        if not keep_archive:
            os.remove(archive)

    # 2. Chunk each article (.nxml) into snippets, one jsonl file per article.
    chunk_dir = os.path.join(destination, "chunk")
    os.makedirs(chunk_dir, exist_ok=True)
    for file_name in sorted(os.listdir(raw_dir)):
        if not file_name.endswith(".nxml"):
            continue
        try:
            snippets = chunk_statpearls_article(os.path.join(raw_dir, file_name))
        except ET.ParseError:
            continue          # a handful of files in the dump are malformed XML
        if not snippets:
            continue
        out_path = os.path.join(chunk_dir, file_name.replace(".nxml", ".jsonl"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(s) for s in snippets))
    return destination


# ----------------------------------------------------------------------------------------
# StatPearls chunker: a port of MedRAG/src/data/statpearls.py (same snippet boundaries)
#
# Rules, in plain words:
#   - every <sec> section is walked paragraph by paragraph;
#   - each snippet's title is "Article -- Section -- Subsection";
#   - a short paragraph (< 200 chars) is glued onto the previous snippet if the result
#     stays under 1000 chars, so we don't create tiny, context-free snippets;
#   - lists are glued on or kept whole when short, split item by item when long.
# ----------------------------------------------------------------------------------------

def _xml_text(element: ET.Element) -> str:
    """All the text inside an XML element, including nested tags, as one string."""
    text = (element.text or "").strip()
    for child in element:
        child_text = _xml_text(child)
        text = (text + " " + child_text) if text else child_text
        if child.tail and child.tail.strip():
            text = (text + " " + child.tail.strip()) if text else child.tail.strip()
    return text.strip()


def _is_subtitle(element: ET.Element) -> bool:
    """A paragraph that is only a <bold> run acts as a subsection heading."""
    children = list(element)
    if element.tag != "p" or len(children) != 1 or children[0].tag != "bold":
        return False
    tail = children[0].tail
    return not (tail and tail.strip())


def chunk_statpearls_article(path: str) -> List[Dict[str, str]]:
    article_id = os.path.basename(path).replace(".nxml", "")
    root = ET.parse(path).getroot()
    title_element = root.find(".//title")
    article_title = (title_element.text or "").strip() if title_element is not None else article_id

    snippets: List[Dict[str, str]] = []

    def start_new_snippet(prefix: str, content: str) -> Dict[str, str]:
        snippet = {"id": f"{article_id}_{len(snippets)}", "title": prefix, "content": content}
        snippets.append(snippet)
        return snippet

    for section in root.findall(".//sec"):
        section_title_element = section.find("./title")
        section_title = ""
        if section_title_element is not None:
            section_title = (section_title_element.text or "").strip()

        prefix = " -- ".join([article_title, section_title])
        last: Optional[Dict[str, str]] = None      # the snippet we may still glue text onto
        previous_element = None

        for element in section:
            if _is_subtitle(element):
                last = None
                subtitle = _xml_text(element)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [subtitle])

            elif element.tag == "p":
                paragraph = _xml_text(element)
                short = len(paragraph) < 200
                if short and last is not None and len(last["content"] + paragraph) < 1000:
                    last["content"] = last["content"] + " " + paragraph
                else:
                    last = start_new_snippet(prefix, paragraph)

            elif element.tag == "list":
                items = [_xml_text(item) for item in element]
                joined = " ".join(items)
                if last is not None and len(joined + last["content"]) < 1000:
                    last["content"] = " ".join([last["content"]] + items)
                elif len(joined) < 1000:
                    last = start_new_snippet(prefix, joined)
                else:
                    last = None
                    for item in items:
                        start_new_snippet(prefix, item)

                # A list right after a subtitle ends that subsection.
                if previous_element is not None and _is_subtitle(previous_element):
                    prefix = " -- ".join([article_title, section_title])

            previous_element = element

    return snippets
