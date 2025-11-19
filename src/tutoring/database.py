"""
Database Manager
Handles data persistence for the tutoring platform
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from .models import (
    Student, Concept, Exercise, Attempt,
    LearningSession, KnowledgeState, MasteryLevel,
    QuestionType, DifficultyLevel
)


class DatabaseManager:
    """
    SQLite database manager for tutoring platform
    """

    def __init__(self, db_path: str = "tutoring_data.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                grade_level INTEGER,
                learning_preferences TEXT,
                total_exercises_completed INTEGER DEFAULT 0,
                total_time_spent INTEGER DEFAULT 0,
                streak_days INTEGER DEFAULT 0,
                achievements TEXT,
                created_at TEXT,
                last_active TEXT
            )
        """)

        # Concepts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                subject TEXT,
                prerequisites TEXT,
                difficulty INTEGER,
                tags TEXT,
                created_at TEXT
            )
        """)

        # Exercises table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id TEXT PRIMARY KEY,
                concept_id TEXT,
                question_type TEXT,
                difficulty INTEGER,
                question TEXT,
                options TEXT,
                correct_answer TEXT,
                explanation TEXT,
                hints TEXT,
                estimated_time INTEGER,
                points INTEGER,
                created_at TEXT,
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
        """)

        # Attempts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY,
                student_id TEXT,
                exercise_id TEXT,
                concept_id TEXT,
                answer TEXT,
                is_correct INTEGER,
                time_spent INTEGER,
                hints_used INTEGER,
                timestamp TEXT,
                confidence REAL,
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id),
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
        """)

        # Knowledge states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_states (
                student_id TEXT,
                concept_id TEXT,
                mastery_probability REAL,
                mastery_level INTEGER,
                attempts INTEGER,
                correct_attempts INTEGER,
                last_practiced TEXT,
                next_review TEXT,
                stability REAL,
                difficulty REAL,
                retrievability REAL,
                PRIMARY KEY (student_id, concept_id),
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
        """)

        # Learning sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                student_id TEXT,
                subject TEXT,
                start_time TEXT,
                end_time TEXT,
                exercises_completed INTEGER,
                correct_answers INTEGER,
                concepts_practiced TEXT,
                total_time INTEGER,
                performance_score REAL,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)

        self.conn.commit()

    # Student operations

    def save_student(self, student: Student) -> None:
        """Save or update student in database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO students
            (id, name, email, grade_level, learning_preferences,
             total_exercises_completed, total_time_spent, streak_days,
             achievements, created_at, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student.id,
            student.name,
            student.email,
            student.grade_level,
            json.dumps(student.learning_preferences),
            student.total_exercises_completed,
            student.total_time_spent,
            student.streak_days,
            json.dumps(student.achievements),
            student.created_at.isoformat(),
            student.last_active.isoformat()
        ))

        # Save knowledge states
        for concept_id, state in student.knowledge_states.items():
            self.save_knowledge_state(student.id, state)

        self.conn.commit()

    def get_student(self, student_id: str) -> Optional[Student]:
        """Retrieve student from database"""
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM students WHERE id = ?",
            (student_id,)
        ).fetchone()

        if not row:
            return None

        student = Student(
            id=row['id'],
            name=row['name'],
            email=row['email'],
            grade_level=row['grade_level'],
            learning_preferences=json.loads(row['learning_preferences']),
            total_exercises_completed=row['total_exercises_completed'],
            total_time_spent=row['total_time_spent'],
            streak_days=row['streak_days'],
            achievements=json.loads(row['achievements']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_active=datetime.fromisoformat(row['last_active'])
        )

        # Load knowledge states
        student.knowledge_states = self.get_knowledge_states(student_id)

        return student

    def get_student_by_email(self, email: str) -> Optional[Student]:
        """Retrieve student by email"""
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM students WHERE email = ?",
            (email,)
        ).fetchone()

        if not row:
            return None

        return self.get_student(row['id'])

    def get_all_students(self) -> List[Student]:
        """Get all students"""
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT id FROM students").fetchall()
        return [self.get_student(row['id']) for row in rows]

    # Concept operations

    def save_concept(self, concept: Concept) -> None:
        """Save concept to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO concepts
            (id, name, description, subject, prerequisites, difficulty, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            concept.id,
            concept.name,
            concept.description,
            concept.subject,
            json.dumps(concept.prerequisites),
            concept.difficulty.value,
            json.dumps(concept.tags),
            concept.created_at.isoformat()
        ))
        self.conn.commit()

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieve concept from database"""
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM concepts WHERE id = ?",
            (concept_id,)
        ).fetchone()

        if not row:
            return None

        return Concept(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            subject=row['subject'],
            prerequisites=json.loads(row['prerequisites']),
            difficulty=DifficultyLevel(row['difficulty']),
            tags=json.loads(row['tags']),
            created_at=datetime.fromisoformat(row['created_at'])
        )

    def get_concepts_by_subject(self, subject: str) -> List[Concept]:
        """Get all concepts for a subject"""
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT id FROM concepts WHERE subject = ?",
            (subject,)
        ).fetchall()
        return [self.get_concept(row['id']) for row in rows]

    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts"""
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT id FROM concepts").fetchall()
        return [self.get_concept(row['id']) for row in rows]

    # Exercise operations

    def save_exercise(self, exercise: Exercise) -> None:
        """Save exercise to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO exercises
            (id, concept_id, question_type, difficulty, question, options,
             correct_answer, explanation, hints, estimated_time, points, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exercise.id,
            exercise.concept_id,
            exercise.question_type.value,
            exercise.difficulty.value,
            exercise.question,
            json.dumps(exercise.options),
            json.dumps(exercise.correct_answer),
            exercise.explanation,
            json.dumps(exercise.hints),
            exercise.estimated_time,
            exercise.points,
            exercise.created_at.isoformat()
        ))
        self.conn.commit()

    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        """Retrieve exercise from database"""
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM exercises WHERE id = ?",
            (exercise_id,)
        ).fetchone()

        if not row:
            return None

        return Exercise(
            id=row['id'],
            concept_id=row['concept_id'],
            question_type=QuestionType(row['question_type']),
            difficulty=DifficultyLevel(row['difficulty']),
            question=row['question'],
            options=json.loads(row['options']),
            correct_answer=json.loads(row['correct_answer']),
            explanation=row['explanation'],
            hints=json.loads(row['hints']),
            estimated_time=row['estimated_time'],
            points=row['points'],
            created_at=datetime.fromisoformat(row['created_at'])
        )

    def get_exercises_by_concept(self, concept_id: str) -> List[Exercise]:
        """Get all exercises for a concept"""
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT id FROM exercises WHERE concept_id = ?",
            (concept_id,)
        ).fetchall()
        return [self.get_exercise(row['id']) for row in rows]

    # Attempt operations

    def save_attempt(self, attempt: Attempt) -> None:
        """Save attempt to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO attempts
            (id, student_id, exercise_id, concept_id, answer, is_correct,
             time_spent, hints_used, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attempt.id,
            attempt.student_id,
            attempt.exercise_id,
            attempt.concept_id,
            json.dumps(attempt.answer),
            1 if attempt.is_correct else 0,
            attempt.time_spent,
            attempt.hints_used,
            attempt.timestamp.isoformat(),
            attempt.confidence
        ))
        self.conn.commit()

    def get_attempts_by_student(
        self,
        student_id: str,
        limit: Optional[int] = None
    ) -> List[Attempt]:
        """Get attempts for a student"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM attempts WHERE student_id = ? ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        rows = cursor.execute(query, (student_id,)).fetchall()

        attempts = []
        for row in rows:
            attempts.append(Attempt(
                id=row['id'],
                student_id=row['student_id'],
                exercise_id=row['exercise_id'],
                concept_id=row['concept_id'],
                answer=json.loads(row['answer']),
                is_correct=bool(row['is_correct']),
                time_spent=row['time_spent'],
                hints_used=row['hints_used'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                confidence=row['confidence']
            ))

        return attempts

    # Knowledge state operations

    def save_knowledge_state(self, student_id: str, state: KnowledgeState) -> None:
        """Save knowledge state"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge_states
            (student_id, concept_id, mastery_probability, mastery_level,
             attempts, correct_attempts, last_practiced, next_review,
             stability, difficulty, retrievability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            state.concept_id,
            state.mastery_probability,
            state.mastery_level.value,
            state.attempts,
            state.correct_attempts,
            state.last_practiced.isoformat() if state.last_practiced else None,
            state.next_review.isoformat() if state.next_review else None,
            state.stability,
            state.difficulty,
            state.retrievability
        ))
        self.conn.commit()

    def get_knowledge_states(self, student_id: str) -> Dict[str, KnowledgeState]:
        """Get all knowledge states for a student"""
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM knowledge_states WHERE student_id = ?",
            (student_id,)
        ).fetchall()

        states = {}
        for row in rows:
            state = KnowledgeState(
                concept_id=row['concept_id'],
                mastery_probability=row['mastery_probability'],
                mastery_level=MasteryLevel(row['mastery_level']),
                attempts=row['attempts'],
                correct_attempts=row['correct_attempts'],
                last_practiced=datetime.fromisoformat(row['last_practiced']) if row['last_practiced'] else None,
                next_review=datetime.fromisoformat(row['next_review']) if row['next_review'] else None,
                stability=row['stability'],
                difficulty=row['difficulty'],
                retrievability=row['retrievability']
            )
            states[state.concept_id] = state

        return states

    # Session operations

    def save_session(self, session: LearningSession) -> None:
        """Save learning session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO learning_sessions
            (id, student_id, subject, start_time, end_time,
             exercises_completed, correct_answers, concepts_practiced,
             total_time, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.student_id,
            session.subject,
            session.start_time.isoformat(),
            session.end_time.isoformat() if session.end_time else None,
            session.exercises_completed,
            session.correct_answers,
            json.dumps(session.concepts_practiced),
            session.total_time,
            session.performance_score
        ))
        self.conn.commit()

    def get_sessions_by_student(
        self,
        student_id: str,
        limit: Optional[int] = None
    ) -> List[LearningSession]:
        """Get sessions for a student"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM learning_sessions WHERE student_id = ? ORDER BY start_time DESC"
        if limit:
            query += f" LIMIT {limit}"

        rows = cursor.execute(query, (student_id,)).fetchall()

        sessions = []
        for row in rows:
            sessions.append(LearningSession(
                id=row['id'],
                student_id=row['student_id'],
                subject=row['subject'],
                start_time=datetime.fromisoformat(row['start_time']),
                end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                exercises_completed=row['exercises_completed'],
                correct_answers=row['correct_answers'],
                concepts_practiced=json.loads(row['concepts_practiced']),
                total_time=row['total_time'],
                performance_score=row['performance_score']
            ))

        return sessions

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
