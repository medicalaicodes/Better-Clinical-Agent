"""
The medical texts we search, and how we get them onto disk.

We use exactly the same corpora as the MedRAG / MIRAGE paper (Xiong et al., 2024), so our
numbers can be compared with theirs:

  Textbooks   18 USMLE textbooks, about 125,800 snippets. Downloaded ready-made from the
              Hugging Face dataset "MedRAG/textbooks".
  StatPearls  about 9,300 clinical articles, about 301,000 snippets. Not on Hugging Face, so we
              download NCBI's archive (~2 GB) and cut every article into snippets ourselves,
              using the same rules as MedRAG's own chunker.

On disk every corpus is a folder of .jsonl files, one snippet per line:
    <corpus_dir>/<name>/chunk/*.jsonl      {"id": ..., "title": ..., "content": ...}

Important: a snippet's position (its "row") in the CorpusStore is also its number inside the
BM25 index and the FAISS index. So the order must never change between building and loading.
That's why the whole corpus is saved once as a single corpus.jsonl next to the indexes.
"""

import json
import os
import tarfile
import xml.etree.ElementTree as ET

STATPEARLS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"


class Snippet:
    """One searchable piece of text."""

    def __init__(self, id, title, content, source):
        self.id = id              # e.g. "First_Aid_Step2_496" or "article-12345_3"
        self.title = title        # book name, or "Article -- Section -- Subsection" for StatPearls
        self.content = content
        self.source = source      # "textbooks" or "statpearls"

        # What we actually index: the title often says what the paragraph is about
        # ("Pulmonary Embolism -- Evaluation") even when the paragraph itself never does.
        if title:
            self.text = title + ". " + content
        else:
            self.text = content


class CorpusStore:
    """All snippets, kept as four plain lists instead of 427k objects (much less memory)."""

    def __init__(self, snippets):
        self.ids = []
        self.titles = []
        self.contents = []
        self.sources = []
        for snippet in snippets:
            self.ids.append(snippet.id)
            self.titles.append(snippet.title)
            self.contents.append(snippet.content)
            self.sources.append(snippet.source)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, row):
        # corpus[row] gives back a Snippet, built on the fly from the four lists.
        return Snippet(self.ids[row], self.titles[row], self.contents[row], self.sources[row])

    def texts(self):
        all_texts = []
        for row in range(len(self.ids)):
            all_texts.append(self[row].text)
        return all_texts

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in range(len(self.ids)):
                record = {"id": self.ids[row], "title": self.titles[row],
                          "content": self.contents[row], "source": self.sources[row]}
                f.write(json.dumps(record) + "\n")


def load_corpus_store(path):
    snippets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            snippets.append(Snippet(record["id"], record["title"], record["content"], record["source"]))
    return CorpusStore(snippets)


# ----------------------------------------------------------------------------------------
# Reading the downloaded corpora
# ----------------------------------------------------------------------------------------

def read_chunk_dir(chunk_dir, source):
    """Every snippet from a folder of .jsonl files. Files are read in sorted order, so the
    snippet order (and therefore the row numbers) is the same on every machine."""
    snippets = []
    for file_name in sorted(os.listdir(chunk_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        with open(os.path.join(chunk_dir, file_name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                snippets.append(Snippet(record["id"], record.get("title", ""), record["content"], source))
    return snippets


def load_corpora(root, names):
    snippets = []
    for name in names:
        chunk_dir = os.path.join(root, name, "chunk")
        if not os.path.isdir(chunk_dir):
            raise FileNotFoundError(f"Corpus '{name}' not found at {chunk_dir}. "
                                    f"Run: python scripts/build_index.py download --corpora {name}")
        snippets += read_chunk_dir(chunk_dir, name)
    return CorpusStore(snippets)


# ----------------------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------------------

def download_textbooks(root):
    """MedRAG/textbooks from Hugging Face: already cut into snippets, about 210 MB."""
    from huggingface_hub import snapshot_download

    destination = os.path.join(root, "textbooks")
    snapshot_download(repo_id="MedRAG/textbooks", repo_type="dataset",
                      allow_patterns=["chunk/*"], local_dir=destination)
    return destination


def download_statpearls(root, keep_archive=False):
    """Downloads the StatPearls archive from NCBI (~2 GB), unpacks it, and chunks every article."""
    import requests
    from tqdm import tqdm

    destination = os.path.join(root, "statpearls")
    os.makedirs(destination, exist_ok=True)
    archive = os.path.join(destination, "statpearls_NBK430685.tar.gz")
    raw_dir = os.path.join(destination, "statpearls_NBK430685")

    # 1. Download and unpack. We write to a ".part" file and rename it only when the download
    #    finished, so a half-downloaded file is never mistaken for a complete one.
    if not os.path.isdir(raw_dir):
        if not os.path.exists(archive):
            with requests.get(STATPEARLS_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                bar = tqdm(total=total, unit="B", unit_scale=True, desc="StatPearls")
                with open(archive + ".part", "wb") as f:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        f.write(block)
                        bar.update(len(block))
                bar.close()
            os.replace(archive + ".part", archive)

        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter="data")    # "data" blocks files escaping the folder
        if not keep_archive:
            os.remove(archive)

    # 2. Cut each article (.nxml file) into snippets, one .jsonl file per article.
    chunk_dir = os.path.join(destination, "chunk")
    os.makedirs(chunk_dir, exist_ok=True)
    for file_name in sorted(os.listdir(raw_dir)):
        if not file_name.endswith(".nxml"):
            continue
        try:
            snippets = chunk_statpearls_article(os.path.join(raw_dir, file_name))
        except ET.ParseError:
            continue          # a handful of files in the dump are broken XML; we skip them
        if not snippets:
            continue
        out_path = os.path.join(chunk_dir, file_name.replace(".nxml", ".jsonl"))
        with open(out_path, "w", encoding="utf-8") as f:
            lines = []
            for snippet in snippets:
                lines.append(json.dumps(snippet))
            f.write("\n".join(lines))
    return destination


# ----------------------------------------------------------------------------------------
# The StatPearls chunker - same rules as MedRAG's statpearls.py, so snippet boundaries match:
#   - every <sec> section is walked paragraph by paragraph;
#   - each snippet's title is "Article -- Section -- Subsection";
#   - a short paragraph (under 200 characters) is glued onto the previous snippet if the
#     result stays under 1,000 characters, so we don't get tiny snippets with no context;
#   - a list is glued on or kept whole when short, and split item by item when long.
# ----------------------------------------------------------------------------------------

def xml_text(element):
    """All the text inside an XML element, including text in nested tags, as one string."""
    text = (element.text or "").strip()
    for child in element:
        child_text = xml_text(child)
        if text:
            text = text + " " + child_text
        else:
            text = child_text
        if child.tail and child.tail.strip():
            if text:
                text = text + " " + child.tail.strip()
            else:
                text = child.tail.strip()
    return text.strip()


def is_subtitle(element):
    """A paragraph that is nothing but one bold run works as a subsection heading."""
    children = list(element)
    if element.tag != "p" or len(children) != 1 or children[0].tag != "bold":
        return False
    tail = children[0].tail
    if tail and tail.strip():
        return False
    return True


def chunk_statpearls_article(path):
    article_id = os.path.basename(path).replace(".nxml", "")
    root = ET.parse(path).getroot()
    title_element = root.find(".//title")
    if title_element is not None:
        article_title = (title_element.text or "").strip()
    else:
        article_title = article_id

    snippets = []

    for section in root.findall(".//sec"):
        section_title = ""
        section_title_element = section.find("./title")
        if section_title_element is not None:
            section_title = (section_title_element.text or "").strip()

        prefix = article_title + " -- " + section_title
        last = None                 # the snippet we may still glue short text onto
        previous_element = None

        for element in section:
            if is_subtitle(element):
                last = None
                subtitle = xml_text(element)
                prefix = article_title + " -- " + section_title + " -- " + subtitle

            elif element.tag == "p":
                paragraph = xml_text(element)
                is_short = len(paragraph) < 200
                if is_short and last is not None and len(last["content"] + paragraph) < 1000:
                    last["content"] = last["content"] + " " + paragraph
                else:
                    last = {"id": f"{article_id}_{len(snippets)}", "title": prefix, "content": paragraph}
                    snippets.append(last)

            elif element.tag == "list":
                items = []
                for item in element:
                    items.append(xml_text(item))
                joined = " ".join(items)

                if last is not None and len(joined + last["content"]) < 1000:
                    last["content"] = last["content"] + " " + joined
                elif len(joined) < 1000:
                    last = {"id": f"{article_id}_{len(snippets)}", "title": prefix, "content": joined}
                    snippets.append(last)
                else:
                    last = None
                    for item in items:
                        snippets.append({"id": f"{article_id}_{len(snippets)}", "title": prefix, "content": item})

                # A list right after a subtitle closes that subsection.
                if previous_element is not None and is_subtitle(previous_element):
                    prefix = article_title + " -- " + section_title

            previous_element = element

    return snippets
