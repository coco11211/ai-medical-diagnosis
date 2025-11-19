"""
Knowledge Tracing System
Implements Bayesian Knowledge Tracing (BKT) to track student knowledge state
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from .models import Student, Attempt, KnowledgeState, MasteryLevel, Concept


class KnowledgeTracer:
    """
    Bayesian Knowledge Tracing implementation
    Tracks probability that a student has mastered each concept
    """

    def __init__(
        self,
        p_learn: float = 0.3,      # Probability of learning from practice
        p_guess: float = 0.25,     # Probability of guessing correctly
        p_slip: float = 0.1,       # Probability of making a careless mistake
        p_init: float = 0.0,       # Initial knowledge probability
        mastery_threshold: float = 0.95
    ):
        """
        Initialize BKT parameters

        Args:
            p_learn: Learning rate (probability of transitioning from not-learned to learned)
            p_guess: Probability of answering correctly without knowing
            p_slip: Probability of answering incorrectly despite knowing
            p_init: Initial probability of knowing the concept
            mastery_threshold: Threshold for considering concept mastered
        """
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip
        self.p_init = p_init
        self.mastery_threshold = mastery_threshold

    def update_knowledge(
        self,
        student: Student,
        attempt: Attempt
    ) -> KnowledgeState:
        """
        Update student's knowledge state based on an attempt using BKT

        Args:
            student: Student object
            attempt: Attempt object

        Returns:
            Updated KnowledgeState
        """
        concept_id = attempt.concept_id

        # Get or create knowledge state
        if concept_id not in student.knowledge_states:
            student.knowledge_states[concept_id] = KnowledgeState(
                concept_id=concept_id,
                mastery_probability=self.p_init
            )

        state = student.knowledge_states[concept_id]

        # Current probability of knowing the concept
        p_know = state.mastery_probability

        # Update based on attempt result
        if attempt.is_correct:
            # P(K|correct) = P(K) * (1 - P(slip)) / P(correct)
            # P(correct) = P(K) * (1 - P(slip)) + (1 - P(K)) * P(guess)
            p_correct = p_know * (1 - self.p_slip) + (1 - p_know) * self.p_guess
            p_know_given_correct = (p_know * (1 - self.p_slip)) / p_correct if p_correct > 0 else p_know
            p_know = p_know_given_correct
        else:
            # P(K|incorrect) = P(K) * P(slip) / P(incorrect)
            # P(incorrect) = P(K) * P(slip) + (1 - P(K)) * (1 - P(guess))
            p_incorrect = p_know * self.p_slip + (1 - p_know) * (1 - self.p_guess)
            p_know_given_incorrect = (p_know * self.p_slip) / p_incorrect if p_incorrect > 0 else p_know
            p_know = p_know_given_incorrect

        # Apply learning (probability increases with practice)
        p_know = p_know + (1 - p_know) * self.p_learn

        # Clamp to [0, 1]
        p_know = max(0.0, min(1.0, p_know))

        # Update state
        state.mastery_probability = p_know
        state.attempts += 1
        if attempt.is_correct:
            state.correct_attempts += 1
        state.last_practiced = datetime.now()

        # Update mastery level
        state.mastery_level = self._get_mastery_level(p_know, state.attempts)

        return state

    def _get_mastery_level(self, probability: float, attempts: int) -> MasteryLevel:
        """Determine mastery level based on probability and attempts"""
        if probability >= 0.95:
            return MasteryLevel.MASTERED
        elif probability >= 0.75:
            return MasteryLevel.PROFICIENT
        elif probability >= 0.5:
            return MasteryLevel.FAMILIAR
        elif attempts > 0:
            return MasteryLevel.LEARNING
        else:
            return MasteryLevel.NOT_LEARNED

    def predict_performance(
        self,
        student: Student,
        concept_id: str
    ) -> float:
        """
        Predict probability of correct answer on next attempt

        Args:
            student: Student object
            concept_id: Concept ID

        Returns:
            Probability of success (0-1)
        """
        if concept_id not in student.knowledge_states:
            # Never practiced - use guess probability
            return self.p_guess

        state = student.knowledge_states[concept_id]
        p_know = state.mastery_probability

        # P(correct) = P(know) * (1 - P(slip)) + (1 - P(know)) * P(guess)
        p_correct = p_know * (1 - self.p_slip) + (1 - p_know) * self.p_guess

        return p_correct

    def get_weak_concepts(
        self,
        student: Student,
        min_attempts: int = 3,
        max_mastery: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Identify concepts where student is struggling

        Args:
            student: Student object
            min_attempts: Minimum attempts to consider
            max_mastery: Maximum mastery probability to consider weak

        Returns:
            List of (concept_id, mastery_probability) tuples
        """
        weak_concepts = []

        for concept_id, state in student.knowledge_states.items():
            if state.attempts >= min_attempts and state.mastery_probability < max_mastery:
                weak_concepts.append((concept_id, state.mastery_probability))

        # Sort by mastery probability (weakest first)
        weak_concepts.sort(key=lambda x: x[1])

        return weak_concepts

    def get_ready_for_review(
        self,
        student: Student,
        concepts: List[Concept]
    ) -> List[str]:
        """
        Get concepts that are ready for review based on spaced repetition

        Args:
            student: Student object
            concepts: List of available concepts

        Returns:
            List of concept IDs ready for review
        """
        ready = []
        now = datetime.now()

        for concept in concepts:
            if concept.id in student.knowledge_states:
                state = student.knowledge_states[concept.id]

                # Check if review is due
                if state.next_review and now >= state.next_review:
                    ready.append(concept.id)

        return ready

    def is_concept_mastered(
        self,
        student: Student,
        concept_id: str
    ) -> bool:
        """
        Check if student has mastered a concept

        Args:
            student: Student object
            concept_id: Concept ID

        Returns:
            True if mastered, False otherwise
        """
        if concept_id not in student.knowledge_states:
            return False

        state = student.knowledge_states[concept_id]
        return state.mastery_probability >= self.mastery_threshold

    def get_overall_mastery(self, student: Student) -> float:
        """
        Calculate overall mastery across all concepts

        Args:
            student: Student object

        Returns:
            Overall mastery score (0-1)
        """
        if not student.knowledge_states:
            return 0.0

        total_mastery = sum(
            state.mastery_probability
            for state in student.knowledge_states.values()
        )

        return total_mastery / len(student.knowledge_states)

    def get_mastery_by_subject(
        self,
        student: Student,
        concepts: List[Concept]
    ) -> Dict[str, float]:
        """
        Calculate mastery grouped by subject

        Args:
            student: Student object
            concepts: List of all concepts

        Returns:
            Dictionary mapping subject to mastery score
        """
        subject_mastery = {}
        subject_counts = {}

        for concept in concepts:
            if concept.id in student.knowledge_states:
                state = student.knowledge_states[concept.id]
                subject = concept.subject

                if subject not in subject_mastery:
                    subject_mastery[subject] = 0.0
                    subject_counts[subject] = 0

                subject_mastery[subject] += state.mastery_probability
                subject_counts[subject] += 1

        # Average mastery per subject
        for subject in subject_mastery:
            if subject_counts[subject] > 0:
                subject_mastery[subject] /= subject_counts[subject]

        return subject_mastery
