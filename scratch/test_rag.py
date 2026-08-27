import urllib.request, urllib.parse, json, re

def search_exact_statpearls(condition: str):
    # Strip non-alphanumeric noise
    clean_cond = re.sub(r"[/()\-]", " ", condition).strip()
    words = [w for w in clean_cond.split() if w.lower() not in ["with", "and", "or", "secondary", "to", "in"]]
    
    # Try search terms in order of specificity
    queries = [
        f'"{clean_cond}"[Title] AND statpearls[book]',
        f'{" ".join(words[:2])}[Title] AND statpearls[book]',
        f'{words[0]}[Title] AND statpearls[book]',
    ]
    if len(words) > 1:
        queries.append(f'{words[-1]}[Title] AND statpearls[book]')

    for q in queries:
        params = {
            "db": "books",
            "term": q,
            "retmode": "json",
            "retmax": "10",
            "tool": "HERALD_ClinicalAgent",
            "email": "clinical_ai@iitmandi.ac.in",
        }
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "HERALD-Medical-Agent/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            uids = data.get("esearchresult", {}).get("idlist", [])
            if not uids:
                continue

            sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=books&id={','.join(uids)}&retmode=json"
            req2 = urllib.request.Request(sum_url, headers={"User-Agent": "HERALD-Medical-Agent/2.0"})
            with urllib.request.urlopen(req2, timeout=4) as resp2:
                sdata = json.loads(resp2.read().decode("utf-8"))

            best_acc = None
            best_title = None
            best_overlap = 0

            target_words = set(w.lower() for w in words)
            for uid in uids:
                info = sdata.get("result", {}).get(uid, {})
                title = info.get("title", "")
                acc = info.get("accessionid") or info.get("chapteraccessionid")
                rtype = info.get("rtype", "")
                
                # Check keyword overlap with title
                title_words = set(re.findall(r"\w+", title.lower()))
                overlap = len(target_words.intersection(title_words))
                
                if overlap > best_overlap and acc and rtype in ["chapter", "book", ""]:
                    best_overlap = overlap
                    best_acc = acc
                    best_title = title

            if best_acc:
                return best_acc, best_title
        except Exception as e:
            continue

    return "", ""

for q in ["Community-Acquired Bacterial Pneumonia", "Influenza with Secondary Bacterial Pneumonia", "Empyema / Parapneumonic Effusion", "Pulmonary Embolism", "Heart Failure"]:
    acc, title = search_exact_statpearls(q)
    print(f"Query: {q:<45} -> Accession: {acc:<12} | Title: {title}")
