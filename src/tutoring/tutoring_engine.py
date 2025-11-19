"""
Main Tutoring Engine
Orchestrates all components of the AI tutoring platform
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
import time

from .models import (
    Student, Concept, Exercise, Attempt,
    LearningSession, QuestionType, DifficultyLevel
)
from .knowledge_tracer import KnowledgeTracer
from .adaptive_learning import AdaptiveLearningEngine
from .exercise_generator import ExerciseGenerator
from .progress_tracker import ProgressTracker
from .database import DatabaseManager
from .notifications import NotificationManager, AchievementSystem


class TutoringEngine:
    """
    Main engine for the AI tutoring platform
    Coordinates all subsystems and provides high-level API
    """

    def __init__(self, db_path: str = "tutoring_data.db"):
        """
        Initialize tutoring engine

        Args:
            db_path: Path to database file
        """
        # Initialize subsystems
        self.database = DatabaseManager(db_path)
        self.knowledge_tracer = KnowledgeTracer()
        self.adaptive_engine = AdaptiveLearningEngine()
        self.exercise_generator = ExerciseGenerator()
        self.progress_tracker = ProgressTracker()
        self.notifier = NotificationManager()
        self.achievement_system = AchievementSystem(self.notifier)

        # Current session
        self.current_session: Optional[LearningSession] = None

    # Student Management

    def create_student(
        self,
        name: str,
        email: str,
        grade_level: int = 1
    ) -> Student:
        """
        Create a new student

        Args:
            name: Student name
            email: Student email
            grade_level: Grade level

        Returns:
            Created Student object
        """
        student = Student(
            name=name,
            email=email,
            grade_level=grade_level
        )

        self.database.save_student(student)
        return student

    def get_student(self, student_id: str) -> Optional[Student]:
        """Get student by ID"""
        return self.database.get_student(student_id)

    def get_student_by_email(self, email: str) -> Optional[Student]:
        """Get student by email"""
        return self.database.get_student_by_email(email)

    # Concept Management

    def create_concept(
        self,
        name: str,
        description: str,
        subject: str,
        difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
        prerequisites: List[str] = None
    ) -> Concept:
        """
        Create a new learning concept

        Args:
            name: Concept name
            description: Concept description
            subject: Subject area
            difficulty: Difficulty level
            prerequisites: List of prerequisite concept IDs

        Returns:
            Created Concept object
        """
        concept = Concept(
            name=name,
            description=description,
            subject=subject,
            difficulty=difficulty,
            prerequisites=prerequisites or []
        )

        self.database.save_concept(concept)
        return concept

    def get_concepts_by_subject(self, subject: str) -> List[Concept]:
        """Get all concepts for a subject"""
        return self.database.get_concepts_by_subject(subject)

    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts"""
        return self.database.get_all_concepts()

    # Learning Sessions

    def start_learning_session(
        self,
        student_id: str,
        subject: str
    ) -> LearningSession:
        """
        Start a new learning session

        Args:
            student_id: Student ID
            subject: Subject to study

        Returns:
            LearningSession object
        """
        student = self.get_student(student_id)
        if not student:
            raise ValueError(f"Student {student_id} not found")

        session = self.progress_tracker.start_session(student, subject)
        self.current_session = session

        # Send notification
        self.notifier.notify_session_start(student.name, subject)

        return session

    def end_learning_session(self, session_id: Optional[str] = None) -> LearningSession:
        """
        End a learning session

        Args:
            session_id: Optional session ID (uses current if None)

        Returns:
            Completed LearningSession object
        """
        session = self.current_session if session_id is None else None

        if not session:
            raise ValueError("No active session")

        # End session and calculate metrics
        session = self.progress_tracker.end_session(session)

        # Save to database
        self.database.save_session(session)

        # Get student and update
        student = self.get_student(session.student_id)

        # Send notification
        if session.exercises_completed > 0:
            accuracy = session.correct_answers / session.exercises_completed
            self.notifier.notify_session_end(
                student.name,
                session.exercises_completed,
                accuracy,
                session.total_time
            )

        # Clear current session
        if self.current_session and self.current_session.id == session.id:
            self.current_session = None

        return session

    # Exercise Generation and Practice

    def get_next_exercise(
        self,
        student_id: str,
        subject: Optional[str] = None
    ) -> Optional[Exercise]:
        """
        Get the next optimal exercise for a student

        Args:
            student_id: Student ID
            subject: Optional subject filter

        Returns:
            Next Exercise or None
        """
        student = self.get_student(student_id)
        if not student:
            return None

        # Get available concepts
        if subject:
            concepts = self.get_concepts_by_subject(subject)
        else:
            concepts = self.get_all_concepts()

        if not concepts:
            return None

        # Use adaptive engine to select concept
        concept_id = self._select_next_concept(student, concepts)
        if not concept_id:
            return None

        concept = self.database.get_concept(concept_id)

        # Generate or retrieve exercise
        exercises = self.database.get_exercises_by_concept(concept_id)

        if exercises:
            # Use adaptive engine to select appropriate exercise
            exercise = self.adaptive_engine.select_next_exercise(
                student,
                [concept],
                exercises
            )
        else:
            # Generate new exercise
            exercise = self.exercise_generator.generate_exercise(concept)
            self.database.save_exercise(exercise)

        return exercise

    def _select_next_concept(
        self,
        student: Student,
        concepts: List[Concept]
    ) -> Optional[str]:
        """Select next concept for practice"""
        concept_scores = []

        for concept in concepts:
            # Calculate priority
            priority = self.adaptive_engine._calculate_concept_priority(
                student,
                concept
            )
            concept_scores.append((concept.id, priority))

        if not concept_scores:
            return None

        # Sort by priority
        concept_scores.sort(key=lambda x: x[1], reverse=True)

        return concept_scores[0][0]

    def submit_answer(
        self,
        student_id: str,
        exercise_id: str,
        answer: Any,
        time_spent: int,
        hints_used: int = 0,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Submit an answer and get feedback

        Args:
            student_id: Student ID
            exercise_id: Exercise ID
            answer: Student's answer
            time_spent: Time spent in seconds
            hints_used: Number of hints used
            confidence: Student's confidence (0-1)

        Returns:
            Dictionary with feedback and updated state
        """
        student = self.get_student(student_id)
        exercise = self.database.get_exercise(exercise_id)

        if not student or not exercise:
            return {"error": "Student or exercise not found"}

        # Check answer
        is_correct = self._check_answer(exercise, answer)

        # Create attempt
        attempt = Attempt(
            student_id=student_id,
            exercise_id=exercise_id,
            concept_id=exercise.concept_id,
            answer=answer,
            is_correct=is_correct,
            time_spent=time_spent,
            hints_used=hints_used,
            confidence=confidence
        )

        # Save attempt
        self.database.save_attempt(attempt)

        # Update knowledge state
        knowledge_state = self.knowledge_tracer.update_knowledge(student, attempt)

        # Update adaptive learning schedule
        rating = self.adaptive_engine.get_performance_rating(
            attempt,
            exercise.estimated_time
        )
        next_review = self.adaptive_engine.schedule_next_review(
            student,
            exercise.concept_id,
            rating
        )

        # Record in progress tracker
        if self.current_session:
            self.progress_tracker.record_attempt(
                student,
                attempt,
                self.current_session
            )

        # Save updated student state
        self.database.save_student(student)

        # Check for achievements
        new_achievements = self.achievement_system.check_achievements(
            student,
            attempt
        )

        # Check for mastery
        if knowledge_state.mastery_probability >= 0.95:
            concept = self.database.get_concept(exercise.concept_id)
            if concept:
                self.notifier.notify_mastery(student.name, concept.name)

        # Return feedback
        feedback = {
            "correct": is_correct,
            "explanation": exercise.explanation,
            "correct_answer": exercise.correct_answer if not is_correct else None,
            "mastery_probability": knowledge_state.mastery_probability,
            "mastery_level": knowledge_state.mastery_level.name,
            "next_review": next_review.isoformat() if next_review else None,
            "points_earned": exercise.points if is_correct else 0,
            "new_achievements": new_achievements
        }

        return feedback

    def _check_answer(self, exercise: Exercise, answer: Any) -> bool:
        """Check if answer is correct"""
        correct = exercise.correct_answer

        # Handle different answer types
        if exercise.question_type == QuestionType.MULTIPLE_CHOICE:
            return str(answer).strip().lower() == str(correct).strip().lower()
        elif exercise.question_type == QuestionType.TRUE_FALSE:
            return str(answer).strip().lower() == str(correct).strip().lower()
        elif exercise.question_type == QuestionType.MATH:
            try:
                return abs(float(answer) - float(correct)) < 0.01
            except:
                return str(answer).strip() == str(correct).strip()
        else:
            # For other types, do string comparison (can be enhanced)
            return str(answer).strip().lower() == str(correct).strip().lower()

    # Progress and Analytics

    def get_progress_report(self, student_id: str) -> Dict[str, Any]:
        """
        Get comprehensive progress report for a student

        Args:
            student_id: Student ID

        Returns:
            Progress report dictionary
        """
        student = self.get_student(student_id)
        if not student:
            return {"error": "Student not found"}

        concepts = self.get_all_concepts()
        sessions = self.database.get_sessions_by_student(student_id)

        report = self.progress_tracker.generate_progress_report(
            student,
            concepts,
            sessions
        )

        return report

    def get_performance_summary(self, student_id: str, days: int = 7) -> Dict:
        """Get performance summary for recent period"""
        student = self.get_student(student_id)
        if not student:
            return {"error": "Student not found"}

        return self.progress_tracker.get_performance_summary(student, days)

    # Recommendations

    def get_study_recommendations(self, student_id: str) -> List[str]:
        """
        Get personalized study recommendations

        Args:
            student_id: Student ID

        Returns:
            List of recommendations
        """
        report = self.get_progress_report(student_id)
        if "error" in report:
            return []

        return report.get("recommendations", [])

    def get_review_concepts(self, student_id: str) -> List[Concept]:
        """
        Get concepts that are due for review

        Args:
            student_id: Student ID

        Returns:
            List of concepts ready for review
        """
        student = self.get_student(student_id)
        if not student:
            return []

        concepts = self.get_all_concepts()
        ready_ids = self.knowledge_tracer.get_ready_for_review(student, concepts)

        return [c for c in concepts if c.id in ready_ids]

    # Bulk Operations

    def generate_exercises_for_concept(
        self,
        concept_id: str,
        count: int = 10,
        question_types: Optional[List[QuestionType]] = None
    ) -> List[Exercise]:
        """
        Generate multiple exercises for a concept

        Args:
            concept_id: Concept ID
            count: Number of exercises to generate
            question_types: Types of questions (all if None)

        Returns:
            List of generated exercises
        """
        concept = self.database.get_concept(concept_id)
        if not concept:
            return []

        exercises = self.exercise_generator.generate_batch(
            concept,
            count,
            question_types
        )

        # Save to database
        for exercise in exercises:
            self.database.save_exercise(exercise)

        return exercises

    def close(self):
        """Close database connection and cleanup"""
        if self.current_session:
            self.end_learning_session()

        self.database.close()
