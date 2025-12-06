"""
Streamlit app for manually comparing anonymization results.

Usage:
    streamlit run compare_anonymization.py
"""

import streamlit as st
import re
from pathlib import Path
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Anonymization Comparison Tool",
    page_icon="🔍",
    layout="wide"
)

PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")


def extract_placeholders_with_context(text: str, context_len: int = 20) -> List[dict]:
    """Extract all placeholders with their surrounding context."""
    placeholders = []
    for m in PLACEHOLDER_RE.finditer(text):
        start, end = m.start(), m.end()
        before = text[max(0, start - context_len):start]
        after = text[end:min(len(text), end + context_len)]
        placeholders.append({
            'label': m.group(),
            'before': before,
            'after': after,
            'start': start,
            'end': end
        })
    return placeholders


def match_placeholders_by_context(pred_phs: List[dict], ref_phs: List[dict]) -> dict:
    """
    Match placeholders based on surrounding context (not exact position).
    Returns dict with matched, fp, fn categorization.
    """
    matched = []  # (pred_idx, ref_idx, label_match)
    pred_matched = set()
    ref_matched = set()
    
    # Try to match each pred placeholder with ref placeholders by context
    for i, pred in enumerate(pred_phs):
        best_match = None
        best_score = 0
        
        for j, ref in enumerate(ref_phs):
            if j in ref_matched:
                continue
            
            # Score based on context similarity
            before_match = pred['before'] == ref['before']
            after_match = pred['after'] == ref['after']
            score = (2 if before_match else 0) + (1 if after_match else 0)
            
            if score > best_score:
                best_score = score
                best_match = j
        
        # If we found a reasonable match
        if best_match is not None and best_score >= 2:  # At least "before" context matches
            label_match = pred['label'] == ref_phs[best_match]['label']
            matched.append((i, best_match, label_match))
            pred_matched.add(i)
            ref_matched.add(best_match)
    
    # False positives: pred not matched
    fp_indices = [i for i in range(len(pred_phs)) if i not in pred_matched]
    
    # False negatives: ref not matched
    fn_indices = [j for j in range(len(ref_phs)) if j not in ref_matched]
    
    return {
        'matched': matched,
        'fp_indices': fp_indices,
        'fn_indices': fn_indices
    }


def highlight_text_by_context(text: str, placeholders: List[dict], 
                               highlight_type: dict) -> str:
    """
    Highlight text based on placeholder classification.
    highlight_type: dict mapping placeholder index to color category
    """
    result = []
    last_idx = 0
    
    for i, ph in enumerate(placeholders):
        start, end = ph['start'], ph['end']
        label = ph['label']
        
        # Add text before placeholder
        if start > last_idx:
            result.append(text[last_idx:start])
        
        # Get highlight type
        htype = highlight_type.get(i, 'none')
        
        if htype == 'correct':
            # True positive - green
            result.append(f'<span style="background-color: #90EE90; padding: 2px 4px; border-radius: 3px;">{label}</span>')
        elif htype == 'label_mismatch':
            # Label mismatch - yellow
            expected = highlight_type.get(f'{i}_expected', '')
            result.append(f'<span style="background-color: #FFD700; padding: 2px 4px; border-radius: 3px;" title="Expected: {expected}">{label}</span>')
        elif htype == 'fp':
            # False positive - red
            result.append(f'<span style="background-color: #FFB6C1; padding: 2px 4px; border-radius: 3px;" title="False Positive">{label}</span>')
        elif htype == 'fn':
            # False negative - orange
            result.append(f'<span style="background-color: #FFA500; padding: 2px 4px; border-radius: 3px;" title="Missing in prediction">{label}</span>')
        else:
            result.append(label)
        
        last_idx = end
    
    # Add remaining text
    if last_idx < len(text):
        result.append(text[last_idx:])
    
    return ''.join(result)


def compute_line_stats(pred_phs: List[dict], ref_phs: List[dict]) -> dict:
    """Compute TP, FP, FN for a single line based on context matching."""
    matching = match_placeholders_by_context(pred_phs, ref_phs)
    
    tp = len([m for m in matching['matched'] if m[2]])  # Label matches
    label_mismatch = len([m for m in matching['matched'] if not m[2]])  # Label doesn't match
    fp = len(matching['fp_indices'])
    fn = len(matching['fn_indices'])
    
    total_matched = len(matching['matched'])
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'label_mismatch': label_mismatch,
        'total_matched': total_matched,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }


def main():
    st.title("🔍 Anonymization Comparison Tool")
    st.markdown("Compare predicted anonymization with reference gold standard")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        original_file = st.text_input("Original file", "example_data/test.short.in.txt")
        pred_file = st.text_input("Predicted file", "example_data/test.short.pred.txt")
        ref_file = st.text_input("Reference file", "example_data/test.short.ref.txt")
        
        show_only_errors = st.checkbox("Show only lines with errors", value=False)
        show_stats = st.checkbox("Show per-line statistics", value=True)
        show_original = st.checkbox("Show original text", value=True)
        
        st.markdown("---")
        st.markdown("### Legend")
        st.markdown("🟢 **Green**: Correct (TP)")
        st.markdown("🟡 **Yellow**: Wrong label")
        st.markdown("🔴 **Pink**: False positive")
        st.markdown("🟠 **Orange**: False negative (missing)")
    
    # Load files
    try:
        original_path = Path(original_file)
        pred_path = Path(pred_file)
        ref_path = Path(ref_file)
        
        if not pred_path.exists() or not ref_path.exists() or not original_path.exists():
            st.error(f"Files not found! Make sure {original_file}, {pred_file} and {ref_file} exist.")
            return
        
        with open(original_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_lines = f.readlines()
        
        with open(pred_path, 'r', encoding='utf-8', errors='ignore') as f:
            pred_lines = f.readlines()
        
        with open(ref_path, 'r', encoding='utf-8', errors='ignore') as f:
            ref_lines = f.readlines()
        
        if len(pred_lines) != len(ref_lines) or len(pred_lines) != len(original_lines):
            st.warning(f"Files have different lengths: {len(original_lines)} vs {len(pred_lines)} vs {len(ref_lines)}")
            num_lines = min(len(original_lines), len(pred_lines), len(ref_lines))
        else:
            num_lines = len(pred_lines)
        
        st.success(f"Loaded {num_lines} lines from both files")
        
        # Overall statistics
        st.header("📊 Overall Statistics")
        total_tp = total_fp = total_fn = 0
        error_lines = []
        
        for i in range(num_lines):
            pred_phs = extract_placeholders_with_context(pred_lines[i])
            ref_phs = extract_placeholders_with_context(ref_lines[i])
            
            stats = compute_line_stats(pred_phs, ref_phs)
            total_tp += stats['tp']
            total_fp += stats['fp']
            total_fn += stats['fn']
            
            if stats['fp'] > 0 or stats['fn'] > 0 or stats.get('label_mismatch', 0) > 0:
                error_lines.append(i)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("True Positives", total_tp)
        with col2:
            st.metric("False Positives", total_fp)
        with col3:
            st.metric("False Negatives", total_fn)
        with col4:
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            st.metric("Precision", f"{precision:.2%}")
        with col5:
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            st.metric("Recall", f"{recall:.2%}")
        
        # Line navigation
        st.header("📄 Line-by-Line Comparison")
        
        if show_only_errors:
            if error_lines:
                line_options = error_lines
                st.info(f"Showing {len(error_lines)} lines with errors")
            else:
                st.success("No errors found!")
                return
        else:
            line_options = list(range(num_lines))
        
        # Line selector
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_line_idx = st.selectbox(
                "Select line number",
                line_options,
                format_func=lambda x: f"Line {x + 1}"
            )
        with col2:
            if st.button("Random Error Line") and error_lines:
                import random
                selected_line_idx = random.choice(error_lines)
                st.rerun()
        
        # Display selected line
        if selected_line_idx < num_lines:
            original_line = original_lines[selected_line_idx].strip()
            pred_line = pred_lines[selected_line_idx].strip()
            ref_line = ref_lines[selected_line_idx].strip()
            
            pred_phs = extract_placeholders_with_context(pred_line)
            ref_phs = extract_placeholders_with_context(ref_line)
            
            stats = compute_line_stats(pred_phs, ref_phs)
            matching = match_placeholders_by_context(pred_phs, ref_phs)
            
            if show_stats:
                st.subheader(f"Line {selected_line_idx + 1} Statistics")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("TP", stats['tp'])
                with col2:
                    st.metric("FP", stats['fp'])
                with col3:
                    st.metric("FN", stats['fn'])
                with col4:
                    st.metric("Label ≠", stats.get('label_mismatch', 0))
                with col5:
                    st.metric("Precision", f"{stats['precision']:.2%}")
                with col6:
                    st.metric("Recall", f"{stats['recall']:.2%}")
            
            # Build highlighting info for predicted
            pred_highlight = {}
            for pred_idx, ref_idx, label_match in matching['matched']:
                if label_match:
                    pred_highlight[pred_idx] = 'correct'
                else:
                    pred_highlight[pred_idx] = 'label_mismatch'
                    pred_highlight[f'{pred_idx}_expected'] = ref_phs[ref_idx]['label']
            
            for idx in matching['fp_indices']:
                pred_highlight[idx] = 'fp'
            
            # Build highlighting info for reference
            ref_highlight = {}
            for pred_idx, ref_idx, label_match in matching['matched']:
                if label_match:
                    ref_highlight[ref_idx] = 'correct'
                else:
                    ref_highlight[ref_idx] = 'label_mismatch'
                    ref_highlight[f'{ref_idx}_expected'] = pred_phs[pred_idx]['label']
            
            for idx in matching['fn_indices']:
                ref_highlight[idx] = 'fn'
            
            # Display comparison
            if show_original:
                st.subheader("Original (test.input.txt)")
                st.markdown(f'<div style="padding: 10px; border: 2px solid #aaa; border-radius: 5px; background-color: #fffef0;">{original_line}</div>', unsafe_allow_html=True)
            
            st.subheader("Predicted (out.txt)")
            pred_highlighted = highlight_text_by_context(pred_line, pred_phs, pred_highlight)
            st.markdown(f'<div style="padding: 10px; border: 2px solid #ccc; border-radius: 5px; background-color: #f9f9f9;">{pred_highlighted}</div>', unsafe_allow_html=True)
            
            st.subheader("Reference (test.ref.txt)")
            ref_highlighted = highlight_text_by_context(ref_line, ref_phs, ref_highlight)
            st.markdown(f'<div style="padding: 10px; border: 2px solid #ccc; border-radius: 5px; background-color: #f0f0ff;">{ref_highlighted}</div>', unsafe_allow_html=True)
            
            # Show placeholder differences
            if matching['fp_indices'] or matching['fn_indices'] or any(not m[2] for m in matching['matched']):
                st.subheader("Differences")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if matching['fp_indices']:
                        st.markdown("**False Positives:**")
                        for idx in matching['fp_indices']:
                            ph = pred_phs[idx]
                            st.markdown(f"- `{ph['label']}` after `...{ph['before'][-15:]}`")
                
                with col2:
                    if matching['fn_indices']:
                        st.markdown("**False Negatives (Missing):**")
                        for idx in matching['fn_indices']:
                            ph = ref_phs[idx]
                            st.markdown(f"- `{ph['label']}` after `...{ph['before'][-15:]}`")
                
                with col3:
                    label_mismatches = [(pred_idx, ref_idx) for pred_idx, ref_idx, lm in matching['matched'] if not lm]
                    if label_mismatches:
                        st.markdown("**Label Mismatches:**")
                        for pred_idx, ref_idx in label_mismatches:
                            pred_label = pred_phs[pred_idx]['label']
                            ref_label = ref_phs[ref_idx]['label']
                            st.markdown(f"- Got `{pred_label}`, Expected `{ref_label}`")
    
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

