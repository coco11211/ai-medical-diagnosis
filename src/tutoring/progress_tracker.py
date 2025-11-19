"""
Progress Tracker
Tracks and analyzes student learning progress and performance
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
from .models import (
    Student, Attempt, LearningSession, Concept,
    KnowledgeState, MasteryLevel
)


class ProgressTracker:
    """
    Comprehensive progress tracking and analytics for students
    """

    def __init__(self):
        """Initialize progress tracker"""
        self.session_cache: Dict[str, LearningSession] = {}

    def start_session(
        self,
        student: Student,
        subject: str
    ) -> LearningSession:
        """
        Start a new learning session

        Args:
            student: Student object
            subject: Subject being studied

        Returns:
            LearningSession object
        """
        session = LearningSession(
            student_id=student.id,
            subject=subject,
            start_time=datetime.now()
        )

        self.session_cache[session.id] = session
        student.last_active = datetime.now()

        return session

    def end_session(
        self,
        session: LearningSession
    ) -> LearningSession:
        """
        End a learning session and calculate metrics

        Args:
            session: LearningSession object

        Returns:
            Updated LearningSession with metrics
        """
        session.end_time = datetime.now()
        session.total_time = int((session.end_time - session.start_time).total_seconds())

        # Calculate performance score
        if session.exercises_completed > 0:
            accuracy = session.correct_answers / session.exercises_completed
            time_efficiency = min(1.0, 1800 / session.total_time)  # Target 30 min
            session.performance_score = (accuracy * 0.7 + time_efficiency * 0.3) * 100

        # Remove from cache
        if session.id in self.session_cache:
            del self.session_cache[session.id]

        return session

    def record_attempt(
        self,
        student: Student,
        attempt: Attempt,
        session: Optional[LearningSession] = None
    ) -> None:
        """
        Record an exercise attempt and update metrics

        Args:
            student: Student object
            attempt: Attempt object
            session: Optional LearningSession to update
        """
        # Update student stats
        student.total_exercises_completed += 1
        student.total_time_spent += attempt.time_spent
        student.last_active = datetime.now()

        # Update session if provided
        if session:
            session.exercises_completed += 1
            if attempt.is_correct:
                session.correct_answers += 1

            if attempt.concept_id not in session.concepts_practiced:
                session.concepts_practiced.append(attempt.concept_id)

    def get_performance_summary(
        self,
        student: Student,
        days: int = 7
    ) -> Dict[str, any]:
        """
        Get performance summary for recent period

        Args:
            student: Student object
            days: Number of days to analyze

        Returns:
            Dictionary with performance metrics
        """
        # In production, this would query from database
        # For now, calculate from knowledge states

        total_concepts = len(student.knowledge_states)
        mastered_concepts = sum(
            1 for state in student.knowledge_states.values()
            if state.mastery_level == MasteryLevel.MASTERED
        )

        avg_mastery = (
            sum(state.mastery_probability for state in student.knowledge_states.values()) /
            total_concepts if total_concepts > 0 else 0.0
        )

        total_attempts = sum(
            state.attempts for state in student.knowledge_states.values()
        )
        total_correct = sum(
            state.correct_attempts for state in student.knowledge_states.values()
        )

        accuracy = total_correct / total_attempts if total_attempts > 0 else 0.0

        return {
            "total_concepts": total_concepts,
            "mastered_concepts": mastered_concepts,
            "mastery_percentage": mastered_concepts / total_concepts * 100 if total_concepts > 0 else 0,
            "average_mastery_probability": avg_mastery,
            "total_attempts": total_attempts,
            "accuracy": accuracy,
            "total_exercises_completed": student.total_exercises_completed,
            "total_time_spent_hours": student.total_time_spent / 3600,
            "streak_days": student.streak_days,
            "achievements": student.achievements
        }

    def get_learning_curve(
        self,
        student: Student,
        concept_id: str
    ) -> List[Tuple[datetime, float]]:
        """
        Get learning curve for a specific concept

        Args:
            student: Student object
            concept_id: Concept ID

        Returns:
            List of (timestamp, mastery_probability) tuples
        """
        # In production, this would query historical data
        # For now, return current state
        if concept_id in student.knowledge_states:
            state = student.knowledge_states[concept_id]
            if state.last_practiced:
                return [(state.last_practiced, state.mastery_probability)]

        return []

    def get_subject_breakdown(
        self,
        student: Student,
        concepts: List[Concept]
    ) -> Dict[str, Dict[str, any]]:
        """
        Get performance breakdown by subject

        Args:
            student: Student object
            concepts: List of all concepts

        Returns:
            Dictionary with subject-level metrics
        """
        subject_data = defaultdict(lambda: {
            "total_concepts": 0,
            "practiced_concepts": 0,
            "mastered_concepts": 0,
            "average_mastery": 0.0,
            "total_attempts": 0,
            "correct_attempts": 0
        })

        # Group concepts by subject
        concept_map = {c.id: c for c in concepts}

        for concept_id, state in student.knowledge_states.items():
            if concept_id in concept_map:
                concept = concept_map[concept_id]
                subject = concept.subject

                subject_data[subject]["practiced_concepts"] += 1
                subject_data[subject]["average_mastery"] += state.mastery_probability
                subject_data[subject]["total_attempts"] += state.attempts
                subject_data[subject]["correct_attempts"] += state.correct_attempts

                if state.mastery_level == MasteryLevel.MASTERED:
                    subject_data[subject]["mastered_concepts"] += 1

        # Count total concepts per subject
        for concept in concepts:
            subject_data[concept.subject]["total_concepts"] += 1

        # Calculate averages
        for subject, data in subject_data.items():
            if data["practiced_concepts"] > 0:
                data["average_mastery"] /= data["practiced_concepts"]
                data["accuracy"] = (
                    data["correct_attempts"] / data["total_attempts"]
                    if data["total_attempts"] > 0 else 0
                )
            else:
                data["accuracy"] = 0

        return dict(subject_data)

    def get_weak_areas(
        self,
        student: Student,
        concepts: List[Concept],
        threshold: float = 0.6
    ) -> List[Dict[str, any]]:
        """
        Identify areas where student is struggling

        Args:
            student: Student object
            concepts: List of all concepts
            threshold: Mastery threshold for "weak" concepts

        Returns:
            List of weak areas with details
        """
        concept_map = {c.id: c for c in concepts}
        weak_areas = []

        for concept_id, state in student.knowledge_states.items():
            if (
                state.attempts >= 3 and
                state.mastery_probability < threshold and
                concept_id in concept_map
            ):
                concept = concept_map[concept_id]
                weak_areas.append({
                    "concept": concept.name,
                    "subject": concept.subject,
                    "mastery": state.mastery_probability,
                    "attempts": state.attempts,
                    "accuracy": state.correct_attempts / state.attempts,
                    "last_practiced": state.last_practiced
                })

        # Sort by mastery (weakest first)
        weak_areas.sort(key=lambda x: x["mastery"])

        return weak_areas

    def get_strengths(
        self,
        student: Student,
        concepts: List[Concept],
        threshold: float = 0.9
    ) -> List[Dict[str, any]]:
        """
        Identify areas where student excels

        Args:
            student: Student object
            concepts: List of all concepts
            threshold: Mastery threshold for "strong" concepts

        Returns:
            List of strong areas with details
        """
        concept_map = {c.id: c for c in concepts}
        strengths = []

        for concept_id, state in student.knowledge_states.items():
            if (
                state.mastery_probability >= threshold and
                concept_id in concept_map
            ):
                concept = concept_map[concept_id]
                strengths.append({
                    "concept": concept.name,
                    "subject": concept.subject,
                    "mastery": state.mastery_probability,
                    "attempts": state.attempts,
                    "accuracy": state.correct_attempts / state.attempts if state.attempts > 0 else 0
                })

        # Sort by mastery (strongest first)
        strengths.sort(key=lambda x: x["mastery"], reverse=True)

        return strengths

    def calculate_study_streak(
        self,
        student: Student,
        sessions: List[LearningSession]
    ) -> int:
        """
        Calculate current study streak in days

        Args:
            student: Student object
            sessions: List of learning sessions

        Returns:
            Streak in days
        """
        if not sessions:
            return 0

        # Sort sessions by date
        sessions_sorted = sorted(sessions, key=lambda s: s.start_time, reverse=True)

        streak = 1
        current_date = sessions_sorted[0].start_time.date()

        for session in sessions_sorted[1:]:
            session_date = session.start_time.date()
            days_diff = (current_date - session_date).days

            if days_diff == 1:
                # Consecutive day
                streak += 1
                current_date = session_date
            elif days_diff == 0:
                # Same day, don't increment
                continue
            else:
                # Streak broken
                break

        return streak

    def get_time_distribution(
        self,
        student: Student,
        concepts: List[Concept]
    ) -> Dict[str, int]:
        """
        Get time spent distribution by subject

        Args:
            student: Student object
            concepts: List of all concepts

        Returns:
            Dictionary mapping subject to time in seconds
        """
        # In production, track actual time per concept
        # For now, estimate based on attempts
        concept_map = {c.id: c for c in concepts}
        time_by_subject = defaultdict(int)

        for concept_id, state in student.knowledge_states.items():
            if concept_id in concept_map:
                concept = concept_map[concept_id]
                # Estimate 90 seconds per attempt
                estimated_time = state.attempts * 90
                time_by_subject[concept.subject] += estimated_time

        return dict(time_by_subject)

    def predict_completion_time(
        self,
        student: Student,
        target_concepts: List[str],
        concepts: List[Concept]
    ) -> Optional[datetime]:
        """
        Predict when student will master target concepts

        Args:
            student: Student object
            target_concepts: List of concept IDs to master
            concepts: All available concepts

        Returns:
            Estimated completion datetime
        """
        concept_map = {c.id: c for c in concepts}

        # Calculate average learning rate
        total_mastery_gain = 0
        total_time = 0

        for concept_id, state in student.knowledge_states.items():
            if state.attempts > 0:
                total_mastery_gain += state.mastery_probability
                # Estimate time based on attempts
                total_time += state.attempts * 90  # 90 sec per attempt

        if total_time == 0:
            return None

        # Average mastery gain per hour
        mastery_per_hour = (total_mastery_gain / total_time) * 3600 if total_time > 0 else 0

        # Calculate remaining mastery needed
        remaining_mastery = 0
        for concept_id in target_concepts:
            if concept_id in student.knowledge_states:
                state = student.knowledge_states[concept_id]
                remaining = max(0, 0.95 - state.mastery_probability)
                remaining_mastery += remaining
            else:
                remaining_mastery += 0.95  # Target 95% mastery

        if mastery_per_hour == 0:
            return None

        # Estimate hours needed
        hours_needed = remaining_mastery / mastery_per_hour

        # Add to current time
        completion_time = datetime.now() + timedelta(hours=hours_needed)

        return completion_time

    def generate_progress_report(
        self,
        student: Student,
        concepts: List[Concept],
        sessions: List[LearningSession]
    ) -> Dict[str, any]:
        """
        Generate comprehensive progress report

        Args:
            student: Student object
            concepts: List of all concepts
            sessions: List of learning sessions

        Returns:
            Comprehensive progress report
        """
        summary = self.get_performance_summary(student)
        subject_breakdown = self.get_subject_breakdown(student, concepts)
        weak_areas = self.get_weak_areas(student, concepts)
        strengths = self.get_strengths(student, concepts)
        streak = self.calculate_study_streak(student, sessions)
        time_dist = self.get_time_distribution(student, concepts)

        return {
            "student_name": student.name,
            "report_date": datetime.now().isoformat(),
            "overall_summary": summary,
            "subject_breakdown": subject_breakdown,
            "weak_areas": weak_areas[:5],  # Top 5 weak areas
            "strengths": strengths[:5],  # Top 5 strengths
            "study_streak": streak,
            "time_distribution": time_dist,
            "achievements": student.achievements,
            "recommendations": self._generate_recommendations(
                student, concepts, weak_areas, strengths
            )
        }

    def _generate_recommendations(
        self,
        student: Student,
        concepts: List[Concept],
        weak_areas: List[Dict],
        strengths: List[Dict]
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        if weak_areas:
            top_weak = weak_areas[0]
            recommendations.append(
                f"Focus on improving {top_weak['concept']} - current mastery: {top_weak['mastery']:.1%}"
            )

        if student.streak_days < 3:
            recommendations.append(
                "Try to study every day to build a learning streak!"
            )

        avg_mastery = sum(
            state.mastery_probability
            for state in student.knowledge_states.values()
        ) / len(student.knowledge_states) if student.knowledge_states else 0

        if avg_mastery < 0.5:
            recommendations.append(
                "Consider reviewing fundamentals before moving to advanced topics"
            )

        if strengths:
            recommendations.append(
                f"Great job on {strengths[0]['concept']}! Consider teaching others to reinforce your knowledge"
            )

        return recommendations
