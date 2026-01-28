import re
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import wikipedia
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Enhanced Hallucination Detection",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'claim_history' not in st.session_state:
    st.session_state.claim_history = []

# Multi-Evidence Sources
def fetch_wikipedia_summary(query: str, sentences: int = 3) -> List[str]:
    """Fast Wikipedia summary"""
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return [s.strip() for s in summary.split('. ') if s.strip()]
    except:
        return []

def fetch_wikipedia_full_content(query: str) -> List[str]:
    """Full Wikipedia page content"""
    try:
        page = wikipedia.page(query)
        content = page.content[:2000]  # Limit to first 2000 chars
        sentences = [s.strip() for s in content.split('. ') if len(s.strip()) > 20]
        return sentences[:8]  # Max 8 sentences
    except:
        return []

def fetch_wikidata_facts(query: str) -> List[str]:
    """Structured facts from Wikidata"""
    try:
        # Search for entity first
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json"
        response = requests.get(search_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('search'):
                entity_id = data['search'][0]['id']
                
                # Get basic facts
                facts_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&format=json&languages=en"
                facts_response = requests.get(facts_url, timeout=5)
                
                if facts_response.status_code == 200:
                    facts_data = facts_response.json()
                    entity = facts_data['entities'][entity_id]
                    
                    facts = []
                    if 'descriptions' in entity and 'en' in entity['descriptions']:
                        facts.append(entity['descriptions']['en']['value'])
                    
                    # Extract some key claims
                    if 'claims' in entity:
                        claims = entity['claims']
                        # Add inception date, creator, etc.
                        for prop in ['P571', 'P170', 'P112', 'P577']:  # inception, creator, founder, publication date
                            if prop in claims and claims[prop]:
                                try:
                                    value = claims[prop][0]['mainsnak']['datavalue']['value']
                                    if isinstance(value, dict) and 'time' in value:
                                        year = value['time'][:5]  # Extract year
                                        facts.append(f"Date: {year}")
                                except:
                                    continue
                    
                    return facts[:3]  # Max 3 facts
        return []
    except:
        return []

@st.cache_data(show_spinner=False)
def fetch_multi_source_evidence(claim: str) -> Dict[str, List[str]]:
    """Fetch evidence from multiple sources"""
    query = extract_key_entities(claim)
    
    evidence = {
        'wikipedia_summary': fetch_wikipedia_summary(query, 3),
        'wikipedia_full': fetch_wikipedia_full_content(query),
        'wikidata_facts': fetch_wikidata_facts(query)
    }
    
    return evidence

def combine_evidence_sources(evidence_dict: Dict[str, List[str]]) -> List[str]:
    """Combine all evidence sources into single list"""
    combined = []
    
    # Priority: Wikidata facts (most reliable) -> Wikipedia summary -> Full content
    combined.extend(evidence_dict.get('wikidata_facts', []))
    combined.extend(evidence_dict.get('wikipedia_summary', []))
    combined.extend(evidence_dict.get('wikipedia_full', []))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_evidence = []
    for item in combined:
        if item not in seen and len(item.strip()) > 10:
            seen.add(item)
            unique_evidence.append(item)
    
    return unique_evidence[:10]  # Max 10 evidence pieces

def extract_years(text):
    """Extract 4-digit years from text"""
    return re.findall(r'\b(19|20)\d{2}\b', text)

def extract_numbers(text):
    """Extract numbers from text"""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return [float(n) for n in numbers if 1800 <= float(n) <= 2100 or float(n) < 200]

def extract_key_entities(claim):
    """Extract key entities from claims for better search"""
    claim_lower = claim.lower()
    
    entities = {
        'great wall of china': 'Great Wall of China',
        'taj mahal': 'Taj Mahal', 
        'eiffel tower': 'Eiffel Tower',
        'python programming': 'Python (programming language)',
        'linux': 'Linux',
        'google': 'Google',
        'virat kohli': 'Virat Kohli',
        'earth': 'Earth',
        'moon': 'Moon'
    }
    
    for key, entity in entities.items():
        if key in claim_lower:
            return entity
    
    # Extract first few important words
    words = claim.split()
    important_words = []
    skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    
    for word in words[:6]:
        if word.lower() not in skip_words and len(word) > 2:
            important_words.append(word)
        if len(important_words) >= 3:
            break
    
    return ' '.join(important_words) if important_words else claim[:50]

def split_into_sentences(text):
    """Split paragraph into individual sentences"""
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    return sentences[:5]

def analyze_single_sentence(sentence: str, evidence_sources: Dict[str, List[str]]) -> Tuple[str, float]:
    """Analyze single sentence with multi-source evidence"""
    if not sentence.strip():
        return "NOT_ENOUGH_INFO", 35.0
    
    combined_evidence = combine_evidence_sources(evidence_sources)
    
    if not combined_evidence:
        return "NOT_ENOUGH_INFO", 35.0
    
    sentence_lower = sentence.lower().strip()
    sentence_years = extract_years(sentence)
    sentence_numbers = extract_numbers(sentence)
    
    contradiction_score = 0
    support_score = 0
    
    # Source weights
    source_weights = {
        'wikidata_facts': 3.0,      # Highest weight - structured data
        'wikipedia_summary': 2.0,    # Medium weight - curated summary
        'wikipedia_full': 1.0        # Lower weight - full content
    }
    
    for source_type, evidence_list in evidence_sources.items():
        weight = source_weights.get(source_type, 1.0)
        
        for evidence in evidence_list:
            evidence_lower = evidence.lower()
            
            # Year contradiction
            evidence_years = extract_years(evidence)
            if sentence_years and evidence_years:
                for sy in sentence_years:
                    for ey in evidence_years:
                        year_diff = abs(int(sy) - int(ey))
                        if year_diff > 5:
                            contradiction_score += 25 * weight
                        elif year_diff <= 2:
                            support_score += 15 * weight
            
            # Number contradiction
            evidence_numbers = extract_numbers(evidence)
            if sentence_numbers and evidence_numbers:
                for sn in sentence_numbers:
                    for en in evidence_numbers:
                        if abs(sn - en) > max(sn * 0.2, 2):
                            contradiction_score += 20 * weight
                        elif abs(sn - en) <= max(sn * 0.05, 1):
                            support_score += 12 * weight
            
            # Strong contradiction patterns
            strong_negations = ['myth', 'false', 'incorrect', 'untrue', 'wrong', 'hoax', 'impossible']
            if any(neg in evidence_lower for neg in strong_negations):
                contradiction_score += 35 * weight
            
            # Keyword support
            sentence_words = set(sentence_lower.split())
            evidence_words = set(evidence_lower.split())
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            sentence_keywords = sentence_words - common_words
            overlap = len(sentence_keywords & evidence_words)
            if overlap > 0:
                support_score += overlap * 3 * weight
    
    # Decision logic
    if contradiction_score >= 30:
        confidence = min(95.0, 70.0 + contradiction_score / 10)
        return "REFUTES", confidence
    elif support_score >= 20 and contradiction_score < 15:
        confidence = min(90.0, 60.0 + support_score / 10)
        return "SUPPORTS", confidence
    elif support_score > contradiction_score:
        confidence = min(80.0, 50.0 + (support_score - contradiction_score) / 5)
        return "SUPPORTS", confidence
    else:
        return "NOT_ENOUGH_INFO", 45.0

def analyze_paragraph_batch(paragraph: str, evidence_sources: Dict[str, List[str]]) -> Tuple[str, float, List[Dict]]:
    """Analyze paragraph by splitting into sentences for batch processing"""
    sentences = split_into_sentences(paragraph)
    
    if len(sentences) <= 1:
        pred, conf = analyze_single_sentence(paragraph, evidence_sources)
        return pred, conf, [{'sentence': paragraph, 'prediction': pred, 'confidence': conf}]
    
    sentence_results = []
    total_support = 0
    total_refute = 0
    total_nei = 0
    
    for sentence in sentences:
        pred, conf = analyze_single_sentence(sentence, evidence_sources)
        sentence_results.append({
            'sentence': sentence,
            'prediction': pred,
            'confidence': conf
        })
        
        if pred == "SUPPORTS":
            total_support += conf
        elif pred == "REFUTES":
            total_refute += conf
        else:
            total_nei += conf
    
    # Weighted voting
    if total_refute > total_support and total_refute > total_nei:
        final_pred = "REFUTES"
        final_conf = min(95.0, total_refute / len(sentences))
    elif total_support > total_refute and total_support > total_nei:
        final_pred = "SUPPORTS"
        final_conf = min(90.0, total_support / len(sentences))
    else:
        final_pred = "NOT_ENOUGH_INFO"
        final_conf = max(40.0, total_nei / len(sentences))
    
    return final_pred, final_conf, sentence_results

# Main UI
st.title("🧠 Enhanced Hallucination Detection with Multi-Source Evidence")
st.write("✅ Single sentence analysis + Paragraph batch processing with Wikipedia + Wikidata")

tab1, tab2, tab3 = st.tabs(["🔍 Single Sentence", "📊 Batch Paragraphs", "📈 History"])

with tab1:
    st.subheader("Single Sentence Analysis")
    st.write("Analyze one sentence with multi-source evidence (Wikipedia summary + full content + Wikidata facts)")
    
    sentence = st.text_input("Enter a single sentence to verify:", 
                            placeholder="Python was created in 1985 for NASA.")
    
    if st.button("🔍 Analyze Sentence"):
        if sentence.strip():
            with st.spinner("Fetching evidence from multiple sources..."):
                evidence_sources = fetch_multi_source_evidence(sentence)
            
            prediction, confidence = analyze_single_sentence(sentence, evidence_sources)
            
            # Store in history
            st.session_state.claim_history.append({
                'claim': sentence,
                'result': prediction,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Result")
                if prediction == "SUPPORTS":
                    st.success(f"✅ **{prediction}** ({confidence:.1f}%)")
                elif prediction == "REFUTES":
                    st.error(f"❌ **{prediction}** ({confidence:.1f}%)")
                else:
                    st.warning(f"⚠️ **{prediction}** ({confidence:.1f}%)")
            
            with col2:
                st.subheader("📊 Evidence Quality")
                total_evidence = sum(len(ev) for ev in evidence_sources.values())
                st.metric("Total Evidence Pieces", total_evidence)
                st.metric("Sources Used", len([k for k, v in evidence_sources.items() if v]))
            
            # Evidence breakdown
            st.subheader("📄 Evidence Sources")
            
            col_wd, col_ws, col_wf = st.columns(3)
            
            with col_wd:
                st.write("**🏛️ Wikidata Facts**")
                facts = evidence_sources.get('wikidata_facts', [])
                if facts:
                    for i, fact in enumerate(facts, 1):
                        st.write(f"{i}. {fact}")
                else:
                    st.write("No structured facts found")
            
            with col_ws:
                st.write("**📖 Wikipedia Summary**")
                summary = evidence_sources.get('wikipedia_summary', [])
                if summary:
                    for i, sent in enumerate(summary, 1):
                        st.write(f"{i}. {sent}")
                else:
                    st.write("No summary found")
            
            with col_wf:
                st.write("**📚 Wikipedia Full Content**")
                full_content = evidence_sources.get('wikipedia_full', [])
                if full_content:
                    for i, sent in enumerate(full_content[:3], 1):  # Show first 3
                        st.write(f"{i}. {sent[:100]}...")
                else:
                    st.write("No full content found")

with tab2:
    st.subheader("Batch Paragraph Processing")
    st.write("Process multiple paragraphs - each paragraph is split into sentences and analyzed")
    
    batch_input = st.text_area("Enter paragraphs (separate with double line breaks):", 
                              height=200,
                              placeholder="Paragraph 1 with multiple sentences. Another sentence here.\n\nParagraph 2 with different claims. More sentences.")
    
    if st.button("🚀 Process Batch"):
        if batch_input.strip():
            paragraphs = [p.strip() for p in batch_input.split('\n\n') if p.strip()]
            
            results = []
            progress_bar = st.progress(0)
            
            for i, paragraph in enumerate(paragraphs):
                with st.spinner(f"Processing paragraph {i+1}/{len(paragraphs)}..."):
                    evidence_sources = fetch_multi_source_evidence(paragraph)
                    prediction, confidence, sentence_breakdown = analyze_paragraph_batch(paragraph, evidence_sources)
                    
                    results.append({
                        'paragraph': paragraph[:100] + '...' if len(paragraph) > 100 else paragraph,
                        'prediction': prediction,
                        'confidence': confidence,
                        'sentences_count': len(sentence_breakdown),
                        'evidence_sources': len([k for k, v in evidence_sources.items() if v])
                    })
                    
                    progress_bar.progress((i + 1) / len(paragraphs))
            
            # Display results table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Paragraphs", len(results))
            with col2:
                supports = sum(1 for r in results if r['prediction'] == 'SUPPORTS')
                st.metric("Supports", supports)
            with col3:
                refutes = sum(1 for r in results if r['prediction'] == 'REFUTES')
                st.metric("Refutes", refutes)
            with col4:
                avg_conf = sum(r['confidence'] for r in results) / len(results)
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            
            # Export options
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            with col_exp2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )

with tab3:
    st.subheader("Recent History")
    
    if st.session_state.get("claim_history"):
        df_hist = pd.DataFrame(st.session_state.claim_history)
        st.dataframe(df_hist, use_container_width=True)
        
        fig = px.pie(df_hist, names="result", title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No history yet. Run some claims first.")

st.markdown("---")
st.markdown("**Enhanced Multi-Source Hallucination Detection** | Wikipedia + Wikidata + Advanced Analysis")