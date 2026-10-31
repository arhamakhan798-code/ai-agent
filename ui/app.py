"""
ui/app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit Enterprise Demonstration & Admin Dashboard for TechMart RAG.

Pages / Tabs:
1. 💬 Customer Assistant (Clara Chatbot with live RAG provenance & intent badges)
2. 📦 Product Catalog & Sync (Relational DB explorer, filters, trigger sync)
3. 📄 Policy Ingestion & Chunk Inspector (Upload docs, preview chunking, re-index)
4. 🔍 Retrieval Diagnostics (Test queries, inspect cosine distance, confidence scores)

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="TechMart Enterprise RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application database and ChromaDB
from app.config import settings
from app.core.database import init_db, get_db_context
from app.models.sync_state import SyncState
from app.repositories.product_repository import ProductRepository
from app.services.chroma_service import init_chroma, get_status as get_chroma_status, query_policy
from app.services.chunking import get_chunker, preview_chunks
from app.services.document_loader import load_document
from app.services.policy_ingestion import ingest_directory, ingest_document
from app.services.product_sync import ProductSyncService
from app.services.rag_engine import RAGEngine

init_db()
init_chroma()

# ── Custom CSS for Luxury Styling ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0b0f17; color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        background-color: #161f30;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c9a96e !important;
        color: #0b0f17 !important;
        font-weight: 700;
    }
    .product-card {
        background: #131b2a;
        border: 1px solid rgba(201, 169, 110, 0.3);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-intent {
        background: #2563eb;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-score {
        background: #059669;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 11px;
    }
    .source-box {
        background: #111827;
        border-left: 3px solid #c9a96e;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 13px;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=64)
st.sidebar.title("TechMart AI")
st.sidebar.caption("Enterprise Hybrid RAG Platform v2.0")

# Quick Stats
with get_db_context() as db:
    repo = ProductRepository(db)
    prod_count = repo.count()
    chroma_info = get_chroma_status()

st.sidebar.metric("Live Catalog Products", prod_count)
st.sidebar.metric("Policy Chunks Indexed", chroma_info.get("chunk_count", 0))

st.sidebar.divider()
st.sidebar.markdown("### Quick Navigation")
st.sidebar.info(
    "• **Chat**: Test shopping concierge & policy Q&A\n"
    "• **Catalog**: Manage products & sync sources\n"
    "• **Policies**: Ingest & preview document chunks\n"
    "• **Diagnostics**: Inspect similarity retrieval"
)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_catalog, tab_policies, tab_diagnostics = st.tabs([
    "💬 Clara AI Concierge",
    "📦 Product Catalog & Sync",
    "📄 Policy Documents & Chunks",
    "🔍 Retrieval Diagnostics",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Clara AI Concierge
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Clara — TechMart AI Assistant")
    st.caption("Answers product questions accurately using the relational catalog and policies via ChromaDB.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Clara, your TechMart sales and policy assistant. How can I help you today?"}
        ]

    # Quick test prompt buttons
    st.markdown("**Try asking:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Discount on PS5?"):
            st.session_state.temp_prompt = "Can I get any more discounts in sony playstation slim?"
    with col2:
        if st.button("Recommend laptops"):
            st.session_state.temp_prompt = "Can you recommend top laptops for work and school?"
    with col3:
        if st.button("Return window policy"):
            st.session_state.temp_prompt = "What is the return window for electronics and open boxes?"
    with col4:
        if st.button("Shipping fees & times"):
            st.session_state.temp_prompt = "How much does standard and express shipping cost?"

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("intent"):
                st.markdown(f"<span class='badge-intent'>Intent: {msg['intent']}</span>", unsafe_allow_html=True)
            if msg.get("products"):
                st.markdown("##### Recommended Products:")
                for p in msg["products"]:
                    st.markdown(
                        f"""
                        <div class='product-card'>
                            <h4>{p['name']} ({p['brand']})</h4>
                            <p><strong>Price:</strong> ${p['price']} | <strong>Discount:</strong> {p['discount']}% | <strong>Effective Price:</strong> ${p.get('effective_price', p['price'])}</p>
                            <p><strong>Stock:</strong> {p['stock']} available | <strong>Category:</strong> {p['category']}</p>
                            <p style='color: #94a3b8;'>{p.get('description', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            if msg.get("sources"):
                with st.expander("📚 Verified Policy Citations"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"<div class='source-box'><strong>{s['title']}</strong> ({s['section']}) — Relevance: <span class='badge-score'>{s['score']:.2f}</span></div>",
                            unsafe_allow_html=True,
                        )

    # Chat input
    user_input = st.chat_input("Ask a question about products or store policies...")
    prompt_to_send = user_input or st.session_state.pop("temp_prompt", None)

    if prompt_to_send:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # RAG Engine Response
        with st.chat_message("assistant"):
            with st.spinner("Consulting catalog and policy base..."):
                with get_db_context() as db:
                    engine = RAGEngine(db)
                    history_formatted = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = engine.process_query(
                        user_message=prompt_to_send,
                        conversation_history=history_formatted,
                    )

                st.markdown(result["reply"])
                st.markdown(f"<span class='badge-intent'>Intent: {result['intent']}</span>", unsafe_allow_html=True)

                if result.get("products"):
                    st.markdown("##### Recommended Products:")
                    for p in result["products"]:
                        st.markdown(
                            f"""
                            <div class='product-card'>
                                <h4>{p['name']} ({p['brand']})</h4>
                                <p><strong>Price:</strong> ${p['price']} | <strong>Discount:</strong> {p['discount']}% | <strong>Effective Price:</strong> ${p.get('effective_price', p['price'])}</p>
                                <p><strong>Stock:</strong> {p['stock']} available | <strong>Category:</strong> {p['category']}</p>
                                <p style='color: #94a3b8;'>{p.get('description', '')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if result.get("sources"):
                    with st.expander("📚 Verified Policy Citations"):
                        for s in result["sources"]:
                            st.markdown(
                                f"<div class='source-box'><strong>{s['title']}</strong> ({s['section']}) — Relevance: <span class='badge-score'>{s['score']:.2f}</span></div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["reply"],
                    "intent": result["intent"],
                    "products": result.get("products", []),
                    "sources": result.get("sources", []),
                })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Product Catalog & Sync
# ═══════════════════════════════════════════════════════════════════════════════
with tab_catalog:
    st.subheader("Relational Product Catalog")
    st.caption("Guarantees factual consistency. Prices and inventory are stored in relational tables, not vector embeddings.")

    # Sync Controls
    st.markdown("#### Synchronization Controls")
    sync_col1, sync_col2, sync_col3 = st.columns([2, 2, 2])

    with sync_col1:
        sync_source = st.selectbox("Data Source", ["google_sheets", "csv"], index=0)
    with sync_col2:
        force_sync = st.checkbox("Force Re-sync (Bypass Hash Check)", value=False)
    with sync_col3:
        st.write("")
        st.write("")
        if st.button("🚀 Trigger Sync Now", use_container_width=True):
            with st.spinner(f"Synchronizing from {sync_source}..."):
                with get_db_context() as db:
                    sync_svc = ProductSyncService(db)
                    sync_res = sync_svc.sync(source=sync_source, force=force_sync)
                if sync_res.get("status") in ("success", "skipped"):
                    st.success(f"Sync complete: {sync_res}")
                else:
                    st.error(f"Sync failed: {sync_res.get('error')}")

    # Catalog Filters
    st.markdown("#### Explore Products")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with get_db_context() as db:
        repo = ProductRepository(db)
        categories = ["All"] + repo.get_categories()
        brands = ["All"] + repo.get_brands()

        with f_col1:
            selected_cat = st.selectbox("Category", categories)
        with f_col2:
            selected_brand = st.selectbox("Brand", brands)
        with f_col3:
            in_stock = st.checkbox("In Stock Only", value=False)
        with f_col4:
            search_query = st.text_input("Search (name, SKU, specs)", placeholder="e.g. Sony, MacBook")

        cat_filter = None if selected_cat == "All" else selected_cat
        brand_filter = None if selected_brand == "All" else selected_brand

        products = repo.search(
            query=search_query,
            category=cat_filter,
            brand=brand_filter,
            in_stock_only=in_stock,
            limit=100,
        )

        # Render Table
        if products:
            table_data = [
                {
                    "SKU": p.sku,
                    "Name": p.name,
                    "Category": p.category,
                    "Brand": p.brand,
                    "Price ($)": f"${p.price:.2f}",
                    "Discount (%)": f"{p.discount}%",
                    "Final Price ($)": f"${p.effective_price:.2f}",
                    "Stock": p.stock,
                }
                for p in products
            ]
            st.dataframe(table_data, use_container_width=True)
            st.caption(f"Showing {len(products)} matching products.")
        else:
            st.warning("No products found matching the selected filters.")

    # Sync Audit History
    st.markdown("#### Sync Audit History")
    with get_db_context() as db:
        states = db.query(SyncState).all()
        if states:
            audit_data = [
                {
                    "Source": s.source_name,
                    "Status": s.status,
                    "Rows Synced": s.rows_synced,
                    "Last Sync Time": str(s.last_sync_time),
                    "Content Hash": s.content_hash[:12] + "..." if s.content_hash else "-",
                    "Error": s.error_message or "None",
                }
                for s in states
            ]
            st.table(audit_data)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Policy Documents & Chunks
# ═══════════════════════════════════════════════════════════════════════════════
with tab_policies:
    st.subheader("Policy Document Management & Chunk Inspector")
    st.caption("Documents are loaded, split using structure-aware token budgets, and embedded in ChromaDB.")

    col_up, col_idx = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload new policy document (.txt, .md, .pdf, .docx)",
            type=["txt", "md", "pdf", "docx"],
        )
        if uploaded_file is not None:
            save_path = Path("./data/policies") / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved to {save_path}. Ready for ingestion.")

    with col_idx:
        st.write("")
        st.write("")
        if st.button("⚡ Ingest All Policies into ChromaDB", use_container_width=True):
            with st.spinner("Ingesting policy files into ChromaDB..."):
                records = ingest_directory(Path("./data/policies"))
                st.success(f"Ingested {len(records)} files successfully.")

    st.divider()

    # Chunk Inspector
    st.markdown("#### Inspect Document Chunks")
    policy_files = list(Path("./data/policies").glob("*.*"))
    policy_options = [p.name for p in policy_files if p.suffix.lower() in (".txt", ".md", ".pdf", ".docx")]

    if policy_options:
        selected_doc = st.selectbox("Select document to preview chunking", policy_options)
        target_path = Path("./data/policies") / selected_doc

        doc = load_document(target_path)
        chunker = get_chunker()
        chunks = chunker.chunk_document(doc)

        st.info(f"Document **{doc.document_title}** split into **{len(chunks)} chunks** ({sum(c.token_count for c in chunks)} total tokens).")

        for idx, chunk in enumerate(chunks):
            with st.expander(f"Chunk #{idx + 1} | {chunk.token_count} tokens | Section: {chunk.section_title or 'General'}"):
                st.code(chunk.text, language="markdown")
                st.caption(f"Stable ID: `{chunk.chunk_id}` | Source: `{chunk.source_path}`")
    else:
        st.warning("No policy documents found in data/policies/.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Retrieval Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_diagnostics:
    st.subheader("Semantic Retrieval Diagnostics")
    st.caption("Test semantic search against ChromaDB and inspect similarity scores and provenance.")

    diag_query = st.text_input(
        "Enter test policy query",
        value="What is the return window for electronics and laptops?",
    )
    col_k, col_thresh = st.columns(2)
    with col_k:
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=4)
    with col_thresh:
        threshold = st.slider("Score Threshold", min_value=0.0, max_value=1.0, value=0.3)

    if st.button("Run Diagnostic Search"):
        with st.spinner("Querying ChromaDB vector collection..."):
            results = query_policy(
                query_text=diag_query,
                n_results=top_k,
                score_threshold=threshold,
            )

        if results:
            st.success(f"Retrieved {len(results)} chunks above score threshold {threshold}:")
            for idx, r in enumerate(results, 1):
                st.markdown(
                    f"""
                    <div class='product-card'>
                        <div style='display:flex; justify-content:space-between;'>
                            <strong>#{idx} {r['document_title']} &rarr; {r['section_title']}</strong>
                            <span class='badge-score'>Similarity: {r['score']:.4f}</span>
                        </div>
                        <p style='margin-top:8px;'>{r['text']}</p>
                        <small style='color:#94a3b8;'>Chunk ID: {r['chunk_id']} | Source: {r['source_path']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No chunks passed the similarity score threshold.")
