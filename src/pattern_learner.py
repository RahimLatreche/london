# src/pattern_learner.py

from typing import Dict, List, Any
from collections import Counter
from datetime import datetime
from src.feedback_io import (
    load_sensor_feedback, 
    append_sensor_feedback, 
    make_sensor_record,
    load_pattern_feedback,
    append_pattern_feedback
)

class PatternLearner:
    """Simple pattern learner that uses feedback_io for data persistence."""
    
    def __init__(self):
        self.sensor_feedback = load_sensor_feedback()
        self.pattern_feedback = load_pattern_feedback()
    
    def record_match_feedback(self, condition: str, sensor_id: str, is_correct: bool, correction: str = None):
        """Record feedback for a sensor match."""
        record = make_sensor_record(condition, sensor_id, is_correct, correction)
        append_sensor_feedback([record])
        # Refresh our local copy
        self.sensor_feedback = load_sensor_feedback()
    
    def record_missing_pattern(self, pattern: str):
        """Record a pattern that should have been found but wasn't."""
        append_pattern_feedback([pattern.strip().lower()])
        # Refresh our local copy
        self.pattern_feedback = load_pattern_feedback()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about the feedback collected."""
        sensor_fb = self.sensor_feedback
        pattern_fb = self.pattern_feedback
        
        if not sensor_fb:
            return {
                'total_feedback': 0,
                'accuracy': 0,
                'corrections': 0,
                'missing_patterns': len(pattern_fb.get('missing', []))
            }
        
        correct_count = sum(1 for fb in sensor_fb if fb.get('was_correct', False))
        correction_count = sum(1 for fb in sensor_fb if fb.get('correction', '').strip())
        
        return {
            'total_feedback': len(sensor_fb),
            'accuracy': (correct_count / len(sensor_fb)) * 100 if sensor_fb else 0,
            'corrections': correction_count,
            'missing_patterns': len(pattern_fb.get('missing', []))
        }
    
    def get_problematic_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns that have low accuracy or many corrections."""
        pattern_stats = {}
        
        for fb in self.sensor_feedback:
            condition = fb.get('condition', '')
            is_correct = fb.get('was_correct', False)
            has_correction = bool(fb.get('correction', '').strip())
            
            if condition not in pattern_stats:
                pattern_stats[condition] = {
                    'total': 0,
                    'correct': 0,
                    'corrections': 0
                }
            
            pattern_stats[condition]['total'] += 1
            if is_correct:
                pattern_stats[condition]['correct'] += 1
            if has_correction:
                pattern_stats[condition]['corrections'] += 1
        
        # Calculate accuracy and flag problematic patterns
        problematic = []
        for pattern, stats in pattern_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            # Flag as problematic if accuracy < 60% and we have at least 3 feedback records
            if stats['total'] >= 3 and accuracy < 60:
                problematic.append({
                    'pattern': pattern,
                    'accuracy': accuracy,
                    'total_feedback': stats['total'],
                    'corrections': stats['corrections']
                })
        
        return sorted(problematic, key=lambda x: x['accuracy'])
    
    def get_most_requested_patterns(self) -> List[Dict[str, Any]]:
        """Get the most frequently requested missing patterns."""
        missing_patterns = self.pattern_feedback.get('missing', [])
        if not missing_patterns:
            return []
        
        pattern_counts = Counter(missing_patterns)
        return [
            {'pattern': pattern, 'count': count}
            for pattern, count in pattern_counts.most_common(10)
        ]
    
    def generate_report(self) -> str:
        """Generate a simple text report of the feedback data."""
        stats = self.get_feedback_stats()
        problematic = self.get_problematic_patterns()
        requested = self.get_most_requested_patterns()
        
        report = f"""Feedback Summary:
• Total sensor feedback records: {stats['total_feedback']}
• Overall accuracy: {stats['accuracy']:.1f}%
• Corrections provided: {stats['corrections']}
• Missing patterns reported: {stats['missing_patterns']}

"""
        
        if problematic:
            report += "Problematic Patterns (low accuracy):\n"
            for p in problematic[:5]:
                report += f"• {p['pattern']}: {p['accuracy']:.1f}% accuracy ({p['total_feedback']} feedback records)\n"
            report += "\n"
        
        if requested:
            report += "Most Requested Missing Patterns:\n"
            for r in requested[:5]:
                report += f"• '{r['pattern']}' (requested {r['count']} times)\n"
        
        return report