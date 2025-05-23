def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    def boost(w, score):
        scores[w] = max(scores.get(w, 0), score)

    # STEP 1: cleaned-prefix boost → 100
    prefix_len = min(4, len(q_clean))
    if prefix_len > 0:
        for w in vocab:
            if re.sub(r"[^a-z0-9]", "", w).startswith(q_clean[:prefix_len]):
                boost(w, 100)

    # STEP 2: substring match → 95
    for w in vocab:
        if q_clean in re.sub(r"[^a-z0-9]", "", w):
            boost(w, 95)

    # STEP 3: fuzzy cleaned matches
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        ts = fuzz.token_sort_ratio(q_clean, w_clean)
        pr = fuzz.partial_ratio(q_clean, w_clean)
        if ts >= 70:
            boost(w, ts)
        if pr >= 70:
            boost(w, pr)

    # STEP 4: raw fuzzy matches
    for scorer, threshold in [(fuzz.token_sort_ratio, 60), (fuzz.partial_ratio, 60)]:
        for w, sc, _ in process.extract(q, vocab, scorer=scorer, limit=200):
            if sc >= threshold:
                boost(w, sc)

    # STEP 5: bigram-overlap ≥ 0.5
    for w in vocab:
        ov = ngram_overlap(q, w)
        if ov >= 0.5:
            boost(w, int(ov * 100))

    # STEP 6: Levenshtein distance
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                boost(w, 100 - (10 * d))

    # STEP 7: Double-Metaphone phonetic matches
    for code in doublemetaphone(q_clean):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 100)

    # FINAL RANKING
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], -len(kv[0])))

    # REFINED DEMOTION of short/overused words
    short_words = []
    long_words = []
    for w, s in ranked:
        if len(w) <= 2:
            short_words.append((w, s - 15))  # small penalty
        elif len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff:
            short_words.append((w, s - 10))  # slight demotion
        else:
            long_words.append((w, s))

    reranked = sorted(long_words + short_words, key=lambda kv: (-kv[1], -len(kv[0])))

    return [w for w, _ in reranked][:max_results]
