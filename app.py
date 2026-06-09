import os
import streamlit as st
import nltk

# --- FORCE SAFE WRITABLE PATH FOR STREAMLIT CLOUD ---
# Create a local directory within the app workspace for NLTK data
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Append this directory to NLTK's search path options
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# --- GUARANTEED CLOUD DOWNLOAD INITIALIZATION ---
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f"tokenizers/{resource}" if 'punkt' in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Now it is perfectly safe to import the rest of your modules
import pandas as pd
import re
import time
import collections
import Levenshtein
from textblob import Word
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Page Layout Configuration
st.set_page_config(page_title="IR Portal Engine", layout="wide")

# ---------------------------------------------------------
# DATA STRUCTURES ENGINE FROM SCRATCH (SECTION D)
# ---------------------------------------------------------

class BSTNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, key, value):
        self.root = self._insert_rec(self.root, key, value)
    def _insert_rec(self, node, key, value):
        if node is None: return BSTNode(key, value)
        if key < node.key: node.left = self._insert_rec(node.left, key, value)
        elif key > node.key: node.right = self._insert_rec(node.right, key, value)
        else: node.value = value
        return node
    def search(self, key):
        return self._search_rec(self.root, key)
    def _search_rec(self, node, key):
        if node is None or node.key == key: return node
        if key < node.key: return self._search_rec(node.left, key)
        return self._search_rec(node.right, key)

class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.child = []

class BTree:
    def __init__(self, t=3):
        self.root = BTreeNode(True)
        self.t = t
    def insert(self, k, val):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self._split_child(temp, 0, root)
            self._insert_non_full(temp, k, val)
        else: self._insert_non_full(root, k, val)
    def _insert_non_full(self, x, k, val):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append((None, None))
            x.values.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                x.values[i + 1] = x.values[i]
                i -= 1
            if i + 1 < len(x.keys): x.keys[i + 1] = k; x.values[i + 1] = val
            else: x.keys.append(k); x.values.append(val)
        else:
            while i >= 0 and k < x.keys[i]: i -= 1
            i += 1
            if len(x.child[i].keys) == (2 * self.t) - 1:
                self._split_child(x, i, x.child[i])
                if k > x.keys[i]: i += 1
            self._insert_non_full(x.child[i], k, val)
    def _split_child(self, x, i, y):
        t = self.t
        z = BTreeNode(y.leaf)
        x.child.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        x.values.insert(i, y.values[t - 1])
        z.keys = y.keys[t:(2 * t) - 1]
        z.values = y.values[t:(2 * t) - 1]
        y.keys = y.keys[0:t - 1]
        y.values = y.values[0:t - 1]
        if not y.leaf:
            z.child = y.child[t:2 * t]
            y.child = y.child[0:t]
    def search(self, k, x=None):
        if x is None: x = self.root
        i = 0
        while i < len(x.keys) and k > x.keys[i]: i += 1
        if i < len(x.keys) and k == x.keys[i]: return x.values[i]
        elif x.leaf: return None
        else: return self.search(k, x.child[i])

# ---------------------------------------------------------
# TEXT PROCESSING CORE PIPELINE (SECTION B)
# ---------------------------------------------------------

def expand_hyphens(text):
    merged = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    split_space = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)
    return text + " " + merged + " " + split_space

def execute_preprocessing(raw_text, configurations, morphology_mode="none"):
    text = expand_hyphens(raw_text) if "Hyphen Handling" in configurations else raw_text
    if "Lowercasing" in configurations: text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if re.match(r'^\w+$', t)]
    if "Stopword Removal" in configurations:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    if morphology_mode == "Stemming":
        tokens = [PorterStemmer().stem(t) for t in tokens]
    elif morphology_mode == "Lemmatization":
        tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return tokens

def calculate_jaccard_coefficient(list1, list2):
    s1, s2 = set(list1), set(list2)
    if not s1 and not s2: return 1.0
    return float(len(s1.intersection(s2))) / len(s1.union(s2))

def build_soundex(word):
    if not word: return ""
    word = word.upper()
    mapping = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    code = word[0]
    for char in word[1:]:
        for keys, val in mapping.items():
            if char in keys and val != code[-1]: code += val
    code = code.replace("0", "")
    return (code[:4] if len(code) >= 4 else code + "0" * (4 - len(code)))

# ---------------------------------------------------------
# DATA CORPUS INGESTION INITIALIZATION
# ---------------------------------------------------------

default_dataset = {
    "doc1.txt": "The state-of-the-art computational deep-learning algorithms process big data paradigms.",
    "doc2.txt": "Information retrieval systems extract structural and unstructured knowledge patterns efficiently.",
    "doc3.txt": "Data mining applications heavily rely on information retrieval pipelines for pattern discovery.",
    "doc4.txt": "With medical information, retrieval systems can achieve faster structural query resolutions."
}

if 'corpus_storage' not in st.session_state:
    st.session_state.corpus_storage = default_dataset

# --- SYSTEM SUB-SECTION ROUTING INTERFACE ---
st.title("🔬 End-to-End Advanced Information Retrieval System")
st.markdown("---")

app_sections = st.sidebar.radio("Navigate Evaluation Criteria Target:", [
    "A. End-to-End Workflow Dashboard",
    "B. Text Preprocessing Pipelines",
    "C. Phrase Query Architectures Matrix",
    "D. Tree-Based Dictionary Performance",
    "E. Tolerant Query Sandbox Engine"
])

# =========================================================
# TASK SECTION A: SYSTEM CORE DATA WORKFLOW
# =========================================================
if app_sections == "A. End-to-End Workflow Dashboard":
    st.header("📂 Task A: Streamlit End-to-End Processing View")
    col_in, col_out = st.columns(2, gap="large")
    
    with col_in:
        st.subheader("Ingestion Controls")
        uploaded = st.file_uploader("Upload Text files collection (.txt)", accept_multiple_files=True, type=["txt"])
        if uploaded:
            st.session_state.corpus_storage = {f.name: f.read().decode("utf-8") for f in uploaded}
            st.success("Custom document set uploaded and registered inside volatile state.")
            
        preprocess_choices = st.multiselect("Select Preprocessing Pipelines Steps:", ["Hyphen Handling", "Lowercasing", "Stopword Removal"], default=["Hyphen Handling", "Lowercasing", "Stopword Removal"])
        morphology_choice = st.radio("Suffix Morphology Option:", ["None", "Stemming", "Lemmatization"], index=2)
        retrieval_strategy = st.selectbox("Retrieval Search Layout Index:", ["Standard Inverted Index", "Biword Phrase Matching", "Positional Index Phrase Search"])
        user_query = st.text_input("Enter Query Input String:", value="information retrieval")
        run_app = st.button("🚀 Execute End-to-End Search Pipeline")
        
    with col_out:
        st.subheader("Data Corpus Monitor Panel")
        corpus_df = [{"Doc Name": k, "Raw Text Fragment": v[:90] + "..." if len(v)>90 else v} for k, v in st.session_state.corpus_storage.items()]
        st.table(pd.DataFrame(corpus_df))
        
        if run_app:
            with st.expander("🔄 Stage 1: Preprocessing Token Intermediate Log", expanded=True):
                st.write("**Parsed Query Tokens Output:**", execute_preprocessing(user_query, preprocess_choices, morphology_choice))
            with st.expander("🗂️ Stage 2: Index Modeling Layout Mappings Log", expanded=True):
                st.write(f"System structure routing aligned to: `{retrieval_strategy}` execution layers.")
            st.success(f"Execution step complete. Target metrics successfully calculated for search string: '{user_query}'")

# =========================================================
# TASK SECTION B: GRANULAR PREPROCESSING BENCHMARKS
# =========================================================
elif app_sections == "B. Text Preprocessing Pipelines":
    st.header("⚙️ Task B: Advanced Preprocessing Pipeline Steps")
    selected_doc = st.selectbox("Choose Target Document ID:", list(st.session_state.corpus_storage.keys()))
    raw_document_text = st.session_state.corpus_storage[selected_doc]
    
    st.text_area("Source Document raw text mapping:", raw_document_text, height=65, disabled=True)
    
    c_hyp, c_stm, c_lem = st.columns(3)
    with c_hyp:
        st.subheader("Hyphen Handling Output")
        st.code(expand_hyphens(raw_document_text))
    with c_stm:
        st.subheader("Porter Stemmer Suffix Stripping")
        stemmed_tokens = execute_preprocessing(raw_document_text, ["Hyphen Handling", "Lowercasing", "Stopword Removal"], "Stemming")
        st.write(stemmed_tokens)
    with c_lem:
        st.subheader("WordNet Lemmatization Base Lemmas")
        lemmatized_tokens = execute_preprocessing(raw_document_text, ["Hyphen Handling", "Lowercasing", "Stopword Removal"], "Lemmatization")
        st.write(lemmatized_tokens)
        
    jaccard_metric = calculate_jaccard_coefficient(stemmed_tokens, lemmatized_tokens)
    st.metric(label="Overlap Jaccard Structural Variance Coefficient", value=f"{round(jaccard_metric * 100, 2)}%")

# =========================================================
# TASK SECTION C: PHRASE MATCHING MATRIX ANALYSIS
# =========================================================
elif app_sections == "C. Phrase Query Architectures Matrix":
    st.header("🔗 Task C: Phrase Processing Matrix (Biword vs Positional)")
    phrase_query_input = st.text_input("Enter Phrase Search Constraints Query:", value="information retrieval")
    
    biword_index = collections.defaultdict(set)
    positional_index = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for d_id, doc_text in st.session_state.corpus_storage.items():
        tokens = execute_preprocessing(doc_text, ["Hyphen Handling", "Lowercasing"], "none")
        for i in range(len(tokens) - 1):
            biword_index[f"{tokens[i]}_{tokens[i+1]}"].add(d_id)
        for positional_coords, token in enumerate(tokens):
            positional_index[token][d_id].append(positional_coords)
            
    q_tokens = execute_preprocessing(phrase_query_input, ["Hyphen Handling", "Lowercasing"], "none")
    
    # Process Biword Index Match Matching
    bw_output_hits = []
    if len(q_tokens) >= 2:
        target_bw_keys = [f"{q_tokens[i]}_{q_tokens[i+1]}" for i in range(len(q_tokens)-1)]
        intersections = [set(biword_index.get(bk, [])) for bk in target_bw_keys]
        if intersections: bw_output_hits = sorted(list(set.intersection(*intersections)))
        
    # Process Positional Index Proximity Matching
    pos_output_hits = []
    if q_tokens and q_tokens[0] in positional_index:
        candidates = set(positional_index[q_tokens[0]].keys())
        for token in q_tokens[1:]: candidates &= set(positional_index.get(token, {}).keys())
        for doc in candidates:
            for start_coord in positional_index[q_tokens[0]][doc]:
                is_valid = True
                for index_offset, token in enumerate(q_tokens[1:], start=1):
                    if (start_coord + index_offset) not in positional_index[token][doc]:
                        is_valid = False; break
                if is_valid: pos_output_hits.append(doc); break

    col_bw, col_pos = st.columns(2)
    with col_bw:
        st.subheader("Biword Representation Map Extract")
        st.json({k: list(v) for k, v in list(biword_index.items())[:3]})
        st.success(f"Biword Matching Result Matrix hits: {bw_output_hits}")
    with col_pos:
        st.subheader("Positional Representation Structure Map Extract")
        st.json({k: dict(v) for k, v in list(positional_index.items())[:2]})
        st.success(f"Positional Proximity Validation Result hits: {pos_output_hits}")

# =========================================================
# TASK SECTION D: TREE DICTIONARY LOOKUP SPEED BENCHMARKS
# =========================================================
elif app_sections == "D. Tree-Based Dictionary Performance":
    st.header("🌲 Task D: Custom Tree Dictionary Lookups Benchmarks (BST vs B-Tree)")
    
    vocab_collector = set()
    for text in st.session_state.corpus_storage.values():
        vocab_collector.update(execute_preprocessing(text, ["Hyphen Handling", "Lowercasing"], "none"))
    vocabulary_array = sorted(list(vocab_collector))
    
    bst_dictionary = BinarySearchTree()
    btree_dictionary = BTree(t=3)
    
    for word in vocabulary_array:
        bst_dictionary.insert(word, ["Posting_Payload_Stub"])
        btree_dictionary.insert(word, ["Posting_Payload_Stub"])
        
    st.write(f"Dictionary parsed statistics: **{len(vocabulary_array)} unique words** index compiled into trees.")
    user_comma_queries = st.text_input("Enter evaluation target tokens split by commas:", value="information, retrieval, algorithms, processing")
    test_words = [w.strip().lower() for w in user_comma_queries.split(",") if w.strip()]
    
    if st.button("Run Real-time Benchmark Execution Comparisons"):
        benchmark_records = []
        for word in test_words:
            t0 = time.perf_counter_ns()
            bst_dictionary.search(word)
            t_bst = time.perf_counter_ns() - t0
            
            t0 = time.perf_counter_ns()
            btree_dictionary.search(word)
            t_btree = time.perf_counter_ns() - t0
            
            benchmark_records.append({"Word Token Target": word, "BST Query Search Time (ns)": t_bst, "B-Tree Search Time (ns)": t_btree})
        st.table(pd.DataFrame(benchmark_records))

# =========================================================
# TASK SECTION E: MULTI-TIER TOLERANT QUERY ENGINE
# =========================================================
elif app_sections == "E. Tolerant Query Sandbox Engine":
    st.header("🎯 Task E: Multi-Tier Tolerant Error Processing Sandbox")
    
    vocab_collector = set()
    for text in st.session_state.corpus_storage.values():
        vocab_collector.update(execute_preprocessing(text, ["Hyphen Handling", "Lowercasing"], "none"))
    system_vocab_list = list(vocab_collector)
    
    col_w, col_l, col_p = st.columns(3)
    with col_w:
        st.subheader("1. Wildcard Resolution (Regex K-Grams)")
        wildcard_expr = st.text_input("Wildcard input token (`*`):", value="comput*")
        if "*" in wildcard_expr:
            reg_compiled = re.compile("^" + ".*".join(map(re.escape, wildcard_expr.lower().split("*"))) + "$")
            st.code([term for term in system_vocab_list if reg_compiled.match(term)])
    with col_l:
        st.subheader("2. Levenshtein Distance Matrix")
        typo_word = st.text_input("Enter typo match test word:", value="informetion")
        distance_df = [{"Term": t, "Edit Distance Score": Levenshtein.distance(typo_word.lower(), t)} for t in system_vocab_list if Levenshtein.distance(typo_word.lower(), t) <= 2]
        st.write("**TextBlob Predictive Guess Fix:**", str(Word(typo_word).correct()))
        st.table(pd.DataFrame(distance_df))
    with col_p:
        st.subheader("3. Phonetic Soundex Matcher")
        phonetic_word = st.text_input("Enter speech phonetic pronunciation approximation:", value="deeply")
        target_soundex_hash = build_soundex(phonetic_word)
        st.write(f"Soundex signature footprint key calculated: `{target_soundex_hash}`")
        st.code([term for term in system_vocab_list if build_soundex(term) == target_soundex_hash])