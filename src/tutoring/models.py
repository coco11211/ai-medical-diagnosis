"""
Core data models for the AI tutoring platform
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid


class DifficultyLevel(Enum):
    """Exercise difficulty levels"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class QuestionType(Enum):
    """Types of questions/exercises"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_IN_BLANK = "fill_in_blank"
    CODING = "coding"
    MATH = "math"
    ESSAY = "essay"


class MasteryLevel(Enum):
    """Student mastery levels"""
    NOT_LEARNED = 0
    LEARNING = 1
    FAMILIAR = 2
    PROFICIENT = 3
    MASTERED = 4


@dataclass
class Concept:
    """Represents a learning concept or skill"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    subject: str = ""
    prerequisites: List[str] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Exercise:
    """Represents a practice exercise"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    concept_id: str = ""
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    question: str = ""
    options: List[str] = field(default_factory=list)
    correct_answer: Any = None
    explanation: str = ""
    hints: List[str] = field(default_factory=list)
    estimated_time: int = 60  # seconds
    points: int = 10
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Attempt:
    """Records a student's attempt at an exercise"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = ""
    exercise_id: str = ""
    concept_id: str = ""
    answer: Any = None
    is_correct: bool = False
    time_spent: int = 0  # seconds
    hints_used: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5  # Student's confidence (0-1)


@dataclass
class KnowledgeState:
    """Represents student's knowledge state for a concept"""
    concept_id: str = ""
    mastery_probability: float = 0.0  # 0-1
    mastery_level: MasteryLevel = MasteryLevel.NOT_LEARNED
    attempts: int = 0
    correct_attempts: int = 0
    last_practiced: Optional[datetime] = None
    next_review: Optional[datetime] = None
    stability: float = 1.0  # For spaced repetition
    difficulty: float = 0.5  # For spaced repetition (0-1)
    retrievability: float = 1.0  # For spaced repetition


@dataclass
class Student:
    """Represents a student in the tutoring system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    grade_level: int = 1
    knowledge_states: Dict[str, KnowledgeState] = field(default_factory=dict)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    total_exercises_completed: int = 0
    total_time_spent: int = 0  # seconds
    streak_days: int = 0
    achievements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class LearningSession:
    """Represents a learning session"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = ""
    subject: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    exercises_completed: int = 0
    correct_answers: int = 0
    concepts_practiced: List[str] = field(default_factory=list)
    total_time: int = 0  # seconds
    performance_score: float = 0.0


@dataclass
class LearningPath:
    """Represents a personalized learning path"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = ""
    subject: str = ""
    target_concepts: List[str] = field(default_factory=list)
    completed_concepts: List[str] = field(default_factory=list)
    current_concept: Optional[str] = None
    estimated_completion_date: Optional[datetime] = None
    progress_percentage: float = 0.0
