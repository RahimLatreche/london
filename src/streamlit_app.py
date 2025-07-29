import streamlit as st
import pandas as pd
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

try:
    from src.pattern_learner import PatternLearner
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    st.error("Feedback system not available. Please ensure pattern_learner.py and feedback_io.py are in src/ folder.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Sensor Matcher",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'match_history' not in st.session_state:
    st.session_state.match_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# --- Cached Data Loading ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sensor_data():
    """Load sensor data with error handling and caching."""
    try:
        from src.sensor_matcher import DF_MASTER_FULL, DF_META_FULL
        return DF_MASTER_FULL, DF_META_FULL
    except Exception as e:
        logger.error(f"Failed to load sensor data: {str(e)}")
        st.error("Failed to load sensor database. Please check data files.")
        st.stop()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_equipment_types(df_master):
    """Extract unique equipment types from the database."""
    try:
        equipment_values = df_master['Equipment'].dropna().str.strip()
        # Extract common equipment types
        common_types = ['AHU', 'RTU', 'CRAC', 'VAV', 'FCU', 'FAN', 'PUMP', 'CHILLER']
        found_types = []
        for eq_type in common_types:
            if equipment_values.str.contains(eq_type, case=False, na=False).any():
                found_types.append(eq_type)
        return found_types if found_types else ['AHU']  # Default fallback
    except:
        return ['AHU', 'RTU', 'CRAC', 'VAV', 'FCU']  # Fallback list

# --- Helper Functions ---
def validate_input(rule_text: str, equipment: str, top_k: int) -> Tuple[bool, Optional[str]]:
    """Validate user inputs."""
    if not rule_text or len(rule_text.strip()) < 10:
        return False, "Rule description must be at least 10 characters long."
    
    if len(rule_text) > 5000:
        return False, "Rule description must be less than 5000 characters."
    
    if not equipment:
        return False, "Please select an equipment type."
    
    if top_k < 1 or top_k > 20:
        return False, "Top-k must be between 1 and 20."
    
    return True, None

def safe_get_sensor_info(df_master, df_meta, definition_id: str) -> Dict:
    """Safely retrieve sensor information with error handling."""
    try:
        # Find in master database
        mask = df_master['Definition'] == definition_id
        if not mask.any():
            return {
                'Definition': definition_id,
                'Display Name': 'Not Found',
                'Metadata IDs': [],
                'Markers': '',
                'Equipment': ''
            }
        
        row = df_master.loc[mask].iloc[0]
        display_name = row.get('Display Name', 'Unknown')
        
        # Find metadata IDs
        meta_ids = []
        if display_name and display_name != 'Unknown':
            meta_mask = df_meta['navName'] == display_name
            if meta_mask.any():
                meta_ids = df_meta.loc[meta_mask, 'id'].tolist()
        
        return {
            'Definition': definition_id,
            'Display Name': display_name,
            'Metadata IDs': meta_ids,
            'Markers': row.get('Markers', ''),
            'Equipment': row.get('Equipment', '')
        }
    except Exception as e:
        logger.error(f"Error getting sensor info for {definition_id}: {str(e)}")
        return {
            'Definition': definition_id,
            'Display Name': 'Error',
            'Metadata IDs': [],
            'Markers': '',
            'Equipment': ''
        }

def format_results_for_export(candidates: Dict, best: Dict, missing_metadata: Dict, df_master, df_meta) -> pd.DataFrame:
    """Format results into a comprehensive DataFrame for export."""
    rows = []
    
    # Add matched sensors
    for condition, definitions in candidates.items():
        for i, def_id in enumerate(definitions):
            info = safe_get_sensor_info(df_master, df_meta, def_id)
            # Fixed: Check against the actual best value, not just existence
            is_best = (def_id == best.get(condition)) and (best.get(condition) is not None)
            rows.append({
                'Status': 'Matched',
                'Condition': condition,
                'Rank': i + 1,
                'Is Best Match': is_best,
                'Definition': def_id,
                'Display Name': info['Display Name'],
                'Equipment': info['Equipment'],
                'Markers': info['Markers'],
                'Metadata IDs': ', '.join(info['Metadata IDs'][:5])  # Limit to first 5
            })
    
    # Add missing metadata sensors
    for condition, missing_sensors in missing_metadata.items():
        for i, sensor in enumerate(missing_sensors):
            rows.append({
                'Status': 'Missing Metadata',
                'Condition': condition,
                'Rank': i + 1,
                'Is Best Match': False,
                'Definition': sensor['definition'],
                'Display Name': sensor['display_name'],
                'Equipment': sensor['equipment'][:100],  # Truncate long equipment lists
                'Markers': sensor['markers'][:100],
                'Metadata IDs': 'N/A - Not in metadata'
            })
    
    return pd.DataFrame(rows)

# --- Main UI ---
st.title("Sensor Matcher")
st.markdown("Match natural language rules to sensor definitions using semantic search.")

# Load data
df_master, df_meta = load_sensor_data()
equipment_types = get_equipment_types(df_master)

# Show learning status
try:
    from src.sensor_matcher import PATTERNS, DYNAMIC_PATTERNS, STOPWORDS, DYNAMIC_STOPWORDS
    learned_patterns = len(DYNAMIC_PATTERNS) - len(PATTERNS)
    learned_stopwords = len(DYNAMIC_STOPWORDS) - len(STOPWORDS)
    
    if learned_patterns > 0 or learned_stopwords > 0:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Using {learned_patterns} learned patterns and {learned_stopwords} learned stopwords for improved matching")
        with col2:
            if st.button("View Details"):
                with st.expander("Learned Configurations", expanded=True):
                    st.write(f"**Total Patterns:** {len(DYNAMIC_PATTERNS)} ({len(PATTERNS)} base + {learned_patterns} learned)")
                    st.write(f"**Total Stopwords:** {len(DYNAMIC_STOPWORDS)} ({len(STOPWORDS)} base + {learned_stopwords} learned)")
except:
    pass

# --- Sidebar ---
with st.sidebar:
    st.header("Search Parameters")
    
    with st.form(key="matcher_form", clear_on_submit=False):
        rule_text = st.text_area(
            "Rule Description",
            height=150,
            placeholder="Example: Discharge fan is on, outdoor damper is open more than a threshold...",
            help="Describe the rule conditions in natural language.",
            max_chars=5000
        )
        
        col1, col2 = st.columns(2)
        with col1:
            equipment = st.selectbox(
                "Equipment Type",
                options=equipment_types,
                help="Select the equipment category to filter sensors."
            )
        
        with col2:
            top_k = 10
        
        submitted = st.form_submit_button(
            label="Run Matching",
            type="primary",
            use_container_width=True
        )
    
    # History section
    if st.session_state.match_history:
        st.divider()
        st.subheader("Recent Searches")
        for i, item in enumerate(reversed(st.session_state.match_history[-5:])):
            if st.button(
                f"{item['equipment']} - {item['timestamp']}",
                key=f"history_{i}",
                use_container_width=True
            ):
                st.session_state.current_results = item

# --- Main Content Area ---
if submitted:
    # Validate input
    is_valid, error_msg = validate_input(rule_text, equipment, top_k)
    if not is_valid:
        st.error(f"Error: {error_msg}")
    else:
        st.session_state.processing = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import matcher function
            status_text.text("Loading matcher module...")
            progress_bar.progress(20)
            
            from src.sensor_matcher import match_and_choose
            
            # Run matching with timeout
            status_text.text("Analyzing rule text...")
            progress_bar.progress(40)
            
            start_time = time.time()
            # Check if new version of match_and_choose is available
            try:
                # Try the new version first
                candidates, best, missing_metadata = match_and_choose(
                    rule_text=rule_text,
                    equipment=equipment,
                    top_k=top_k,
                    return_missing=True
                )
            except TypeError:
                # Fall back to old version
                candidates, best = match_and_choose(
                    rule_text=rule_text,
                    equipment=equipment,
                    top_k=top_k
                )
                missing_metadata = {}  # No missing metadata in old version
            elapsed_time = time.time() - start_time
            
            progress_bar.progress(80)
            status_text.text("Processing results...")
            
            # Store results
            result_data = {
                'rule_text': rule_text,
                'equipment': equipment,
                'top_k': top_k,
                'candidates': candidates,
                'best': best,
                'missing_metadata': missing_metadata,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'elapsed_time': elapsed_time
            }
            
            st.session_state.current_results = result_data
            st.session_state.match_history.append(result_data)
            
            # Limit history size
            if len(st.session_state.match_history) > 50:
                st.session_state.match_history = st.session_state.match_history[-50:]
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Matching completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Matching failed: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        finally:
            st.session_state.processing = False

# --- Results Display ---
if st.session_state.current_results and not st.session_state.processing:
    results = st.session_state.current_results
    
    # Get missing_metadata with fallback for old results
    missing_metadata = results.get('missing_metadata', {})
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Equipment", results['equipment'])
    with col2:
        st.metric("Conditions Found", len(results['candidates']))
    with col3:
        st.metric("Total Matches", sum(len(v) for v in results['candidates'].values()))
    with col4:
        # Count total missing sensors
        missing_count = sum(len(v) for v in missing_metadata.values())
        st.metric("Missing Metadata", missing_count)
    with col5:
        st.metric("Match Time", f"{results.get('elapsed_time', 0):.2f}s")
    
    # Export options
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Match Results")
    
    # Prepare export data
    export_df = format_results_for_export(
        results['candidates'],
        results['best'],
        missing_metadata,
        df_master,
        df_meta
    )
    
    with col2:
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="Download Full Results",
            data=csv_data,
            file_name=f"sensor_matches_{results['timestamp'].replace(':', '-')}.csv",
            mime="text/csv"
        )
    
    with col3:
        best_only_df = export_df[export_df['Is Best Match']]
        if not best_only_df.empty:
            st.download_button(
                label="Download Best Only",
                data=best_only_df.to_csv(index=False),
                file_name=f"best_matches_{results['timestamp'].replace(':', '-')}.csv",
                mime="text/csv"
            )
    
    # Rule text display
    with st.expander("Rule Text", expanded=True):
        st.text(results['rule_text'])
    
    # Display missing metadata warning if any
    if missing_metadata:
        with st.expander(f"Warning: Sensors found in master but missing from metadata ({sum(len(v) for v in missing_metadata.values())} sensors)", expanded=False):
            for condition, sensors in missing_metadata.items():
                st.markdown(f"**{condition}:**")
                for sensor in sensors:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.code(sensor['definition'])
                    with col2:
                        st.text(f"{sensor['display_name']} | Equipment: {sensor['equipment'][:50]}...")
    
    # Results tabs - Added Feedback and Learning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Best Matches", "All Candidates", "Statistics", "Provide Feedback", "Learning Analytics"])
    
    with tab1:
        # Best matches summary
        best_data = []
        for condition, best_def in results['best'].items():
            if best_def:
                info = safe_get_sensor_info(df_master, df_meta, best_def)
                best_data.append({
                    'Condition': condition,
                    'Definition': best_def,
                    'Display Name': info['Display Name'],
                    'Equipment': info['Equipment'][:50] + '...' if len(info['Equipment']) > 50 else info['Equipment'],
                    'Metadata IDs': len(info['Metadata IDs'])
                })
        
        if best_data:
            best_df = pd.DataFrame(best_data)
            st.dataframe(
                best_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Definition': st.column_config.TextColumn(width="medium"),
                    'Display Name': st.column_config.TextColumn(width="large"),
                    'Metadata IDs': st.column_config.NumberColumn("Meta Count", width="small")
                }
            )
        else:
            st.warning("No best matches found.")
    
    with tab2:
        # Detailed candidates
        for condition, definitions in results['candidates'].items():
            # Show missing count in expander title if any
            missing_for_condition = len(missing_metadata.get(condition, []))
            expander_title = f"**{condition}** ({len(definitions)} matches"
            if missing_for_condition > 0:
                expander_title += f", {missing_for_condition} missing metadata"
            expander_title += ")"
            
            with st.expander(expander_title, expanded=len(results['candidates']) <= 5):
                if not definitions:
                    st.warning("No matches found for this condition.")
                else:
                    # Create detailed view
                    detail_data = []
                    for i, def_id in enumerate(definitions):
                        info = safe_get_sensor_info(df_master, df_meta, def_id)
                        detail_data.append({
                            'Rank': i + 1,
                            'Best': 'Yes' if def_id == results['best'].get(condition) else '',
                            'Definition': def_id,
                            'Display Name': info['Display Name'],
                            'Markers': info['Markers'][:50] + '...' if len(info['Markers']) > 50 else info['Markers'],
                            'Meta IDs': len(info['Metadata IDs'])
                        })
                    
                    detail_df = pd.DataFrame(detail_data)
                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Rank': st.column_config.NumberColumn(width="small"),
                            'Best': st.column_config.TextColumn(width="small"),
                            'Meta IDs': st.column_config.NumberColumn("Metadata Count", width="small")
                        }
                    )
    
    with tab3:
        # Statistics
        st.subheader("Match Statistics")
        
        # Condition coverage
        conditions_with_matches = sum(1 for v in results['candidates'].values() if v)
        total_conditions = len(results['candidates'])
        conditions_with_missing = sum(1 for v in missing_metadata.values() if v)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Condition Coverage",
                f"{conditions_with_matches}/{total_conditions}",
                f"{(conditions_with_matches/total_conditions*100):.1f}%"
            )
        
        with col2:
            avg_matches = sum(len(v) for v in results['candidates'].values()) / len(results['candidates']) if results['candidates'] else 0
            st.metric("Avg Matches per Condition", f"{avg_matches:.1f}")
        
        with col3:
            if missing_metadata:
                metadata_coverage = (sum(len(v) for v in results['candidates'].values()) / 
                                   (sum(len(v) for v in results['candidates'].values()) + 
                                    sum(len(v) for v in missing_metadata.values())) * 100)
                st.metric("Metadata Coverage", f"{metadata_coverage:.1f}%")
            else:
                st.metric("Metadata Coverage", "100%")
        
        # Distribution chart
        if results['candidates']:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            conditions = list(results['candidates'].keys())
            match_counts = [len(v) for v in results['candidates'].values()]
            missing_counts = [len(missing_metadata.get(c, [])) for c in conditions]
            
            # Create stacked bar chart
            x = range(len(conditions))
            bars1 = ax.bar(x, match_counts, label='Matched', color='#2E86AB')
            bars2 = ax.bar(x, missing_counts, bottom=match_counts, label='Missing Metadata', color='#F24236', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.set_ylabel('Number of Sensors')
            ax.set_title('Sensor Matches per Condition')
            ax.legend()
            
            # Add value labels on bars
            for i, (matched, missing) in enumerate(zip(match_counts, missing_counts)):
                if matched > 0:
                    ax.text(i, matched/2, str(matched), ha='center', va='center', color='white', fontweight='bold')
                if missing > 0:
                    ax.text(i, matched + missing/2, str(missing), ha='center', va='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Missing metadata summary
        if missing_metadata:
            st.divider()
            st.subheader("Missing Metadata Analysis")
            
            missing_summary = []
            for condition, sensors in missing_metadata.items():
                missing_summary.append({
                    'Condition': condition,
                    'Missing Count': len(sensors),
                    'Top Missing Sensors': ', '.join([s['definition'] for s in sensors[:3]])
                })
            
            if missing_summary:
                missing_df = pd.DataFrame(missing_summary)
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("Provide Feedback")
        
        if not FEEDBACK_AVAILABLE:
            st.error("Feedback system not available. Please ensure feedback_io.py and pattern_learner.py are installed.")
        else:
            learner = PatternLearner()
            
            st.markdown("Help improve the system by providing feedback on the sensor matches.")
            
            # Feedback for each best match
            if any(results['best'].values()):
                st.markdown("#### Rate Sensor Matches")
                
                for condition, best_def in results['best'].items():
                    if best_def:
                        with st.expander(f"**{condition}** → {best_def}", expanded=False):
                            info = safe_get_sensor_info(df_master, df_meta, best_def)
                            
                            # Show sensor details
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**Display Name:** {info['Display Name']}")
                                st.write(f"**Equipment:** {info['Equipment'][:100]}...")
                                if info['Markers']:
                                    st.write(f"**Markers:** {info['Markers'][:100]}...")
                            
                            with col2:
                                # Feedback form
                                feedback_key = f"feedback_{condition}_{best_def}"
                                
                                feedback = st.radio(
                                    "Is this match correct?",
                                    ["✅ Correct", "❌ Wrong", "🤔 Unsure"],
                                    key=feedback_key,
                                    horizontal=True
                                )
                                
                                correction = ""
                                if "Wrong" in feedback:
                                    correction = st.text_input(
                                        "Correct sensor ID (optional):",
                                        key=f"correction_{condition}_{best_def}",
                                        placeholder="Enter correct sensor ID..."
                                    )
                                
                                if st.button("Submit Feedback", key=f"submit_{condition}_{best_def}", type="primary"):
                                    is_correct = "Correct" in feedback
                                    
                                    try:
                                        learner.record_match_feedback(
                                            condition=condition,
                                            sensor_id=best_def,
                                            is_correct=is_correct,
                                            correction=correction
                                        )
                                        st.success("✅ Thank you! Feedback saved.")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to save feedback: {e}")
            
            else:
                st.info("No sensor matches to provide feedback on.")
            
            # Report missing patterns
            st.divider()
            st.markdown("#### Report Missing Patterns")
            
            with st.expander("Found a sensor pattern that wasn't detected?", expanded=False):
                st.markdown("If you notice sensor patterns in your rule that weren't found, report them here:")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    missing_pattern = st.text_input(
                        "Missing pattern:",
                        key="missing_pattern_input",
                        placeholder="e.g., 'exhaust fan', 'zone temperature'..."
                    )
                
                with col2:
                    st.write("")  # Spacing
                    if st.button("Report Missing", key="report_missing", type="secondary"):
                        if missing_pattern.strip():
                            try:
                                learner.record_missing_pattern(missing_pattern)
                                st.success(f"✅ Reported: '{missing_pattern}'")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to report pattern: {e}")
                        else:
                            st.error("Please enter a pattern")

    # Replace tab5 (Learning Analytics) with this:
    with tab5:
        st.subheader("Feedback Analytics")
        
        if not FEEDBACK_AVAILABLE:
            st.error("Analytics not available. Please ensure feedback_io.py and pattern_learner.py are installed.")
        else:
            learner = PatternLearner()
            stats = learner.get_feedback_stats()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Feedback", stats['total_feedback'])
            
            with col2:
                accuracy_delta = None
                if stats['total_feedback'] > 0:
                    accuracy_delta = f"{stats['accuracy']:.1f}%"
                st.metric("Accuracy", f"{stats['accuracy']:.1f}%", delta=accuracy_delta)
            
            with col3:
                st.metric("Corrections Given", stats['corrections'])
            
            with col4:
                st.metric("Missing Patterns", stats['missing_patterns'])
            
            # Detailed analytics
            if stats['total_feedback'] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Problematic Patterns")
                    problematic = learner.get_problematic_patterns()
                    
                    if problematic:
                        for p in problematic[:5]:
                            st.warning(f"**{p['pattern']}** - {p['accuracy']:.1f}% accuracy ({p['total_feedback']} feedback)")
                    else:
                        st.success("No problematic patterns found!")
                
                with col2:
                    st.markdown("#### Most Requested Missing")
                    requested = learner.get_most_requested_patterns()
                    
                    if requested:
                        for r in requested[:5]:
                            st.info(f"**'{r['pattern']}'** - requested {r['count']} times")
                    else:
                        st.info("No missing patterns reported yet.")
            
            # Generate and show report
            st.divider()
            st.markdown("#### Feedback Report")
            
            report = learner.generate_report()
            st.text(report)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Feedback Data"):
                    try:
                        feedback_data = learner.sensor_feedback
                        if feedback_data:
                            df_feedback = pd.DataFrame(feedback_data)
                            csv_data = df_feedback.to_csv(index=False)
                            st.download_button(
                                label="Download Sensor Feedback CSV",
                                data=csv_data,
                                file_name=f"sensor_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No feedback data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col2:
                if st.button("📋 Export Missing Patterns"):
                    try:
                        missing_patterns = learner.pattern_feedback.get('missing', [])
                        if missing_patterns:
                            df_missing = pd.DataFrame(missing_patterns, columns=['Missing Pattern'])
                            csv_data = df_missing.to_csv(index=False)
                            st.download_button(
                                label="Download Missing Patterns CSV",
                                data=csv_data,
                                file_name=f"missing_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No missing patterns to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col3:
                if st.button("🗑️ Clear All Data", type="secondary"):
                    if st.checkbox("⚠️ I understand this will delete all feedback", key="confirm_clear"):
                        try:
                            # Clear the files by writing empty defaults
                            from src.feedback_io import sensor_fb_file, pattern_fb_file
                            import json
                            
                            sensor_fb_file.write_text(json.dumps([], indent=2))
                            pattern_fb_file.write_text(json.dumps({"missing": []}, indent=2))
                            
                            st.success("✅ All feedback data cleared!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear data: {e}")
            
            # Show recent feedback if available
            if stats['total_feedback'] > 0:
                st.divider()
                st.markdown("#### Recent Feedback")
                
                recent_feedback = learner.sensor_feedback[-10:]  # Last 10
                display_data = []
                
                for fb in reversed(recent_feedback):  # Most recent first
                    display_data.append({
                        'Date': fb.get('timestamp', '').split('T')[0],
                        'Pattern': fb.get('condition', ''),
                        'Sensor': fb.get('definition', ''),
                        'Correct': '✅' if fb.get('was_correct') else '❌',
                        'Correction': fb.get('correction', '')[:50] + '...' if len(fb.get('correction', '')) > 50 else fb.get('correction', '')
                    })
                
                if display_data:
                    df_recent = pd.DataFrame(display_data)
                    st.dataframe(
                        df_recent,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Date': st.column_config.TextColumn(width="small"),
                            'Pattern': st.column_config.TextColumn(width="medium"),
                            'Sensor': st.column_config.TextColumn(width="medium"),
                            'Correct': st.column_config.TextColumn(width="small"),
                            'Correction': st.column_config.TextColumn(width="large")
                        }
                    )