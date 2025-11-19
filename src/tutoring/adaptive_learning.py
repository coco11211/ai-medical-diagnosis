"""
Adaptive Learning Engine
Implements FSRS (Free Spaced Repetition Scheduler) algorithm for optimal learning
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from .models import Student, Concept, Exercise, KnowledgeState, Attempt, DifficultyLevel


class AdaptiveLearningEngine:
    """
    Adaptive learning engine using FSRS algorithm
    Optimizes review scheduling and content difficulty
    """

    def __init__(
        self,
        initial_stability: float = 1.0,
        initial_difficulty: float = 0.5,
        retrievability_threshold: float = 0.9
    ):
        """
        Initialize adaptive learning engine

        Args:
            initial_stability: Initial memory stability (days)
            initial_difficulty: Initial difficulty rating (0-1)
            retrievability_threshold: Target retrievability for reviews
        """
        self.initial_stability = initial_stability
        self.initial_difficulty = initial_difficulty
        self.retrievability_threshold = retrievability_threshold

        # FSRS parameters (optimized for learning)
        self.w = [
            0.4,    # w[0]: initial stability for Again
            0.6,    # w[1]: initial stability for Hard
            2.4,    # w[2]: initial stability for Good
            5.8,    # w[3]: initial stability for Easy
            4.93,   # w[4]: stability multiplier for Again
            0.94,   # w[5]: stability multiplier for Hard
            0.86,   # w[6]: stability multiplier for Good
            0.01,   # w[7]: stability multiplier for Easy
            1.49,   # w[8]: difficulty weight
            0.14,   # w[9]: difficulty decay
            0.94,   # w[10]: difficulty multiplier
            2.18,   # w[11]: retrievability weight
            0.05,   # w[12]: stability increase for Again
            0.34,   # w[13]: stability increase for Hard
            1.26,   # w[14]: stability increase for Good
            4.93,   # w[15]: stability increase for Easy
        ]

    def schedule_next_review(
        self,
        student: Student,
        concept_id: str,
        rating: int  # 1=Again, 2=Hard, 3=Good, 4=Easy
    ) -> datetime:
        """
        Calculate next optimal review time using FSRS algorithm

        Args:
            student: Student object
            concept_id: Concept ID
            rating: Performance rating (1-4)

        Returns:
            Next review datetime
        """
        if concept_id not in student.knowledge_states:
            student.knowledge_states[concept_id] = KnowledgeState(
                concept_id=concept_id,
                stability=self.initial_stability,
                difficulty=self.initial_difficulty,
                retrievability=1.0
            )

        state = student.knowledge_states[concept_id]

        # Calculate time since last review
        if state.last_practiced:
            days_elapsed = (datetime.now() - state.last_practiced).total_seconds() / 86400
        else:
            days_elapsed = 0

        # Calculate current retrievability
        if days_elapsed > 0 and state.stability > 0:
            retrievability = np.power(1 + days_elapsed / (9 * state.stability), -1)
        else:
            retrievability = 1.0

        state.retrievability = retrievability

        # Update difficulty based on rating
        new_difficulty = self._update_difficulty(state.difficulty, rating)
        state.difficulty = new_difficulty

        # Update stability based on rating and retrievability
        new_stability = self._update_stability(
            state.stability,
            state.difficulty,
            retrievability,
            rating
        )
        state.stability = new_stability

        # Calculate next review interval
        interval_days = self._calculate_interval(
            new_stability,
            self.retrievability_threshold
        )

        # Set next review time
        next_review = datetime.now() + timedelta(days=interval_days)
        state.next_review = next_review

        return next_review

    def _update_difficulty(self, current_difficulty: float, rating: int) -> float:
        """Update difficulty rating based on performance"""
        # Difficulty decreases with good performance, increases with poor performance
        difficulty_delta = self.w[8] * (rating - 3)
        new_difficulty = current_difficulty - difficulty_delta

        # Clamp to [0, 1]
        return max(0.0, min(1.0, new_difficulty))

    def _update_stability(
        self,
        current_stability: float,
        difficulty: float,
        retrievability: float,
        rating: int
    ) -> float:
        """Update memory stability based on performance"""
        if rating == 1:  # Again
            new_stability = self.w[0] * np.exp(self.w[4] * difficulty)
        elif rating == 2:  # Hard
            new_stability = current_stability * (1 + self.w[12])
        elif rating == 3:  # Good
            new_stability = current_stability * (1 + self.w[13] * (1 - retrievability))
        else:  # Easy (rating == 4)
            new_stability = current_stability * (1 + self.w[14])

        return max(0.1, new_stability)  # Minimum stability of 0.1 days

    def _calculate_interval(self, stability: float, target_retrievability: float) -> float:
        """Calculate optimal review interval"""
        # I = S * (R^(1/d) - 1) / (9 * (1 - R))
        # Simplified: I = S * ln(target_R) / ln(0.9)
        if target_retrievability >= 1.0:
            return stability * 365  # Max 1 year

        interval = stability * (
            np.power(target_retrievability, -1/9) - 1
        )

        return max(0.1, min(365, interval))  # Clamp to [0.1, 365] days

    def select_next_exercise(
        self,
        student: Student,
        available_concepts: List[Concept],
        available_exercises: List[Exercise]
    ) -> Optional[Exercise]:
        """
        Select the most appropriate next exercise for the student

        Args:
            student: Student object
            available_concepts: List of concepts to choose from
            available_exercises: List of available exercises

        Returns:
            Selected exercise or None
        """
        if not available_concepts or not available_exercises:
            return None

        # Score each concept based on learning priority
        concept_scores = []

        for concept in available_concepts:
            score = self._calculate_concept_priority(student, concept)
            concept_scores.append((concept.id, score))

        # Sort by priority (highest first)
        concept_scores.sort(key=lambda x: x[1], reverse=True)

        # Select exercises from top priority concepts
        for concept_id, _ in concept_scores[:3]:  # Consider top 3 concepts
            # Filter exercises for this concept
            concept_exercises = [
                ex for ex in available_exercises
                if ex.concept_id == concept_id
            ]

            if concept_exercises:
                # Select exercise with appropriate difficulty
                selected = self._select_appropriate_difficulty(
                    student,
                    concept_id,
                    concept_exercises
                )
                if selected:
                    return selected

        return None

    def _calculate_concept_priority(
        self,
        student: Student,
        concept: Concept
    ) -> float:
        """
        Calculate learning priority for a concept

        Higher score = higher priority
        """
        score = 0.0

        if concept.id in student.knowledge_states:
            state = student.knowledge_states[concept.id]

            # Priority factors:
            # 1. Low retrievability (needs review soon)
            if state.retrievability < 0.9:
                score += (1.0 - state.retrievability) * 10

            # 2. Not yet mastered but started
            if state.mastery_probability < 0.95:
                score += (1.0 - state.mastery_probability) * 5

            # 3. Due for review
            if state.next_review and datetime.now() >= state.next_review:
                days_overdue = (datetime.now() - state.next_review).days
                score += min(days_overdue * 2, 20)

        else:
            # New concept - moderate priority
            score += 3.0

            # Boost if prerequisites are met
            prerequisites_met = all(
                prereq_id in student.knowledge_states and
                student.knowledge_states[prereq_id].mastery_probability >= 0.7
                for prereq_id in concept.prerequisites
            )

            if prerequisites_met or not concept.prerequisites:
                score += 5.0

        return score

    def _select_appropriate_difficulty(
        self,
        student: Student,
        concept_id: str,
        exercises: List[Exercise]
    ) -> Optional[Exercise]:
        """
        Select exercise with appropriate difficulty level

        Args:
            student: Student object
            concept_id: Concept ID
            exercises: List of exercises for this concept

        Returns:
            Selected exercise or None
        """
        if not exercises:
            return None

        # Determine target difficulty based on mastery
        if concept_id in student.knowledge_states:
            state = student.knowledge_states[concept_id]
            mastery = state.mastery_probability

            if mastery < 0.3:
                target_difficulty = DifficultyLevel.BEGINNER
            elif mastery < 0.6:
                target_difficulty = DifficultyLevel.INTERMEDIATE
            elif mastery < 0.85:
                target_difficulty = DifficultyLevel.ADVANCED
            else:
                target_difficulty = DifficultyLevel.EXPERT
        else:
            target_difficulty = DifficultyLevel.BEGINNER

        # Find exercises matching target difficulty
        matching = [ex for ex in exercises if ex.difficulty == target_difficulty]

        if matching:
            # Return random exercise from matching set
            return np.random.choice(matching)

        # If no exact match, find closest difficulty
        exercises_sorted = sorted(
            exercises,
            key=lambda ex: abs(ex.difficulty.value - target_difficulty.value)
        )

        return exercises_sorted[0]

    def create_personalized_session(
        self,
        student: Student,
        available_concepts: List[Concept],
        available_exercises: List[Exercise],
        target_duration: int = 1800,  # 30 minutes in seconds
        target_exercises: int = 10
    ) -> List[Exercise]:
        """
        Create a personalized learning session

        Args:
            student: Student object
            available_concepts: Available concepts
            available_exercises: Available exercises
            target_duration: Target session duration in seconds
            target_exercises: Target number of exercises

        Returns:
            List of selected exercises
        """
        selected_exercises = []
        estimated_time = 0

        while (
            len(selected_exercises) < target_exercises and
            estimated_time < target_duration
        ):
            # Select next exercise
            exercise = self.select_next_exercise(
                student,
                available_concepts,
                available_exercises
            )

            if exercise is None:
                break

            selected_exercises.append(exercise)
            estimated_time += exercise.estimated_time

            # Remove from available to avoid duplicates
            available_exercises = [
                ex for ex in available_exercises
                if ex.id != exercise.id
            ]

        return selected_exercises

    def get_performance_rating(
        self,
        attempt: Attempt,
        expected_time: int
    ) -> int:
        """
        Convert attempt into FSRS rating (1-4)

        Args:
            attempt: Attempt object
            expected_time: Expected time for exercise

        Returns:
            Rating (1=Again, 2=Hard, 3=Good, 4=Easy)
        """
        if not attempt.is_correct:
            return 1  # Again

        # Correct answer - determine difficulty based on time and hints
        time_ratio = attempt.time_spent / expected_time if expected_time > 0 else 1.0

        if attempt.hints_used > 2 or time_ratio > 2.0:
            return 2  # Hard
        elif attempt.hints_used > 0 or time_ratio > 1.5:
            return 3  # Good
        else:
            return 4  # Easy
