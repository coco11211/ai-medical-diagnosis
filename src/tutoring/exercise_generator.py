"""
Exercise Generator
Generates various types of educational exercises and questions
"""
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from .models import Exercise, Concept, QuestionType, DifficultyLevel


class ExerciseGenerator:
    """
    Generates exercises across multiple subjects and question types
    """

    def __init__(self):
        """Initialize exercise generator with templates"""
        self.templates = {
            QuestionType.MULTIPLE_CHOICE: self._generate_multiple_choice,
            QuestionType.TRUE_FALSE: self._generate_true_false,
            QuestionType.SHORT_ANSWER: self._generate_short_answer,
            QuestionType.FILL_IN_BLANK: self._generate_fill_in_blank,
            QuestionType.MATH: self._generate_math_problem,
            QuestionType.CODING: self._generate_coding_problem,
        }

        # Subject-specific generators
        self.subject_generators = {
            "mathematics": self._generate_math_exercises,
            "science": self._generate_science_exercises,
            "programming": self._generate_programming_exercises,
            "language": self._generate_language_exercises,
            "history": self._generate_history_exercises,
        }

    def generate_exercise(
        self,
        concept: Concept,
        question_type: Optional[QuestionType] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> Exercise:
        """
        Generate an exercise for a given concept

        Args:
            concept: Concept to generate exercise for
            question_type: Type of question to generate (random if None)
            difficulty: Difficulty level (uses concept difficulty if None)

        Returns:
            Generated Exercise
        """
        if question_type is None:
            question_type = random.choice(list(QuestionType))

        if difficulty is None:
            difficulty = concept.difficulty

        # Use subject-specific generator if available
        if concept.subject.lower() in self.subject_generators:
            generator = self.subject_generators[concept.subject.lower()]
            return generator(concept, question_type, difficulty)

        # Use generic generator
        if question_type in self.templates:
            generator = self.templates[question_type]
            return generator(concept, difficulty)

        # Fallback to multiple choice
        return self._generate_multiple_choice(concept, difficulty)

    def generate_batch(
        self,
        concept: Concept,
        count: int = 10,
        question_types: Optional[List[QuestionType]] = None
    ) -> List[Exercise]:
        """
        Generate multiple exercises for a concept

        Args:
            concept: Concept to generate exercises for
            count: Number of exercises to generate
            question_types: Types of questions to generate (all if None)

        Returns:
            List of generated exercises
        """
        exercises = []

        if question_types is None:
            question_types = list(QuestionType)

        for _ in range(count):
            q_type = random.choice(question_types)
            exercise = self.generate_exercise(concept, q_type)
            exercises.append(exercise)

        return exercises

    # Generic question type generators

    def _generate_multiple_choice(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate a multiple choice question"""
        # This is a template - in production, use AI or pre-defined questions
        question = f"Which of the following best describes {concept.name}?"

        options = [
            f"Correct definition of {concept.name}",
            f"Incorrect definition 1",
            f"Incorrect definition 2",
            f"Incorrect definition 3"
        ]

        random.shuffle(options)
        correct_answer = options[0]

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.MULTIPLE_CHOICE,
            difficulty=difficulty,
            question=question,
            options=options,
            correct_answer=correct_answer,
            explanation=f"{concept.description}",
            estimated_time=60 + difficulty.value * 30,
            points=10 * difficulty.value
        )

    def _generate_true_false(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate a true/false question"""
        is_true = random.choice([True, False])

        if is_true:
            question = f"{concept.description}"
        else:
            question = f"{concept.name} is not related to {concept.subject}"

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.TRUE_FALSE,
            difficulty=difficulty,
            question=question,
            options=["True", "False"],
            correct_answer="True" if is_true else "False",
            explanation=f"The correct answer is {'True' if is_true else 'False'} because {concept.description}",
            estimated_time=30 + difficulty.value * 15,
            points=5 * difficulty.value
        )

    def _generate_short_answer(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate a short answer question"""
        question = f"Explain the concept of {concept.name} in your own words."

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.SHORT_ANSWER,
            difficulty=difficulty,
            question=question,
            options=[],
            correct_answer=concept.description,
            explanation=f"A good answer should include: {concept.description}",
            estimated_time=120 + difficulty.value * 60,
            points=15 * difficulty.value
        )

    def _generate_fill_in_blank(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate a fill-in-the-blank question"""
        # Replace key term with blank
        description = concept.description
        blank_word = concept.name

        question = description.replace(blank_word, "______", 1)
        question = f"Fill in the blank: {question}"

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.FILL_IN_BLANK,
            difficulty=difficulty,
            question=question,
            options=[],
            correct_answer=blank_word,
            explanation=f"The answer is '{blank_word}'",
            estimated_time=45 + difficulty.value * 20,
            points=8 * difficulty.value
        )

    # Subject-specific generators

    def _generate_math_exercises(
        self,
        concept: Concept,
        question_type: QuestionType,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate mathematics exercises"""
        if "algebra" in concept.name.lower():
            return self._generate_algebra_problem(concept, difficulty)
        elif "geometry" in concept.name.lower():
            return self._generate_geometry_problem(concept, difficulty)
        elif "calculus" in concept.name.lower():
            return self._generate_calculus_problem(concept, difficulty)
        else:
            return self._generate_math_problem(concept, difficulty)

    def _generate_math_problem(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate a general math problem"""
        difficulty_multiplier = difficulty.value

        # Generate based on difficulty
        if difficulty == DifficultyLevel.BEGINNER:
            a, b = random.randint(1, 10), random.randint(1, 10)
            operation = random.choice(['+', '-'])
            if operation == '+':
                question = f"What is {a} + {b}?"
                answer = a + b
            else:
                question = f"What is {a} - {b}?"
                answer = a - b
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            a, b = random.randint(10, 50), random.randint(2, 12)
            operation = random.choice(['*', '/'])
            if operation == '*':
                question = f"What is {a} × {b}?"
                answer = a * b
            else:
                a = a * b  # Ensure clean division
                question = f"What is {a} ÷ {b}?"
                answer = a // b
        else:  # ADVANCED or EXPERT
            a, b, c = random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)
            question = f"Solve for x: {a}x + {b} = {c}"
            answer = round((c - b) / a, 2) if a != 0 else "undefined"

        # Generate distractors for multiple choice
        options = [str(answer)]
        for _ in range(3):
            distractor = answer + random.randint(-5, 5)
            if str(distractor) not in options:
                options.append(str(distractor))

        random.shuffle(options)

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.MATH,
            difficulty=difficulty,
            question=question,
            options=options,
            correct_answer=str(answer),
            explanation=f"The correct answer is {answer}",
            estimated_time=60 * difficulty_multiplier,
            points=10 * difficulty_multiplier
        )

    def _generate_algebra_problem(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate algebra problems"""
        if difficulty == DifficultyLevel.BEGINNER:
            a, b = random.randint(1, 10), random.randint(1, 20)
            question = f"Solve for x: x + {a} = {b}"
            answer = b - a
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            a, b, c = random.randint(2, 10), random.randint(1, 20), random.randint(1, 50)
            question = f"Solve for x: {a}x + {b} = {c}"
            answer = round((c - b) / a, 2) if a != 0 else "undefined"
        else:
            a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 20)
            question = f"Solve for x: {a}x² + {b}x + {c} = 0 (round to 2 decimals)"
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                answer = round((-b + discriminant**0.5) / (2*a), 2)
            else:
                answer = "No real solution"

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.MATH,
            difficulty=difficulty,
            question=question,
            options=[],
            correct_answer=str(answer),
            explanation=f"The solution is x = {answer}",
            estimated_time=90 + difficulty.value * 30,
            points=15 * difficulty.value
        )

    def _generate_geometry_problem(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate geometry problems"""
        if difficulty == DifficultyLevel.BEGINNER:
            length, width = random.randint(5, 20), random.randint(5, 20)
            question = f"What is the area of a rectangle with length {length} and width {width}?"
            answer = length * width
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            radius = random.randint(3, 15)
            question = f"What is the area of a circle with radius {radius}? (Use π ≈ 3.14, round to 2 decimals)"
            answer = round(3.14 * radius ** 2, 2)
        else:
            a, b, c = 3, 4, 5  # Pythagorean triple
            scale = random.randint(1, 5)
            a, b = a * scale, b * scale
            question = f"In a right triangle with legs {a} and {b}, what is the length of the hypotenuse?"
            answer = round((a**2 + b**2) ** 0.5, 2)

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.MATH,
            difficulty=difficulty,
            question=question,
            options=[],
            correct_answer=str(answer),
            explanation=f"The answer is {answer}",
            estimated_time=60 + difficulty.value * 30,
            points=12 * difficulty.value
        )

    def _generate_calculus_problem(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate calculus problems"""
        if difficulty == DifficultyLevel.INTERMEDIATE:
            a = random.randint(2, 10)
            question = f"What is the derivative of f(x) = {a}x²?"
            answer = f"{2*a}x"
        else:
            a, b = random.randint(1, 5), random.randint(1, 5)
            question = f"What is the derivative of f(x) = {a}x³ + {b}x?"
            answer = f"{3*a}x² + {b}"

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.MATH,
            difficulty=difficulty,
            question=question,
            options=[],
            correct_answer=answer,
            explanation=f"The derivative is {answer}",
            estimated_time=120 + difficulty.value * 30,
            points=20 * difficulty.value
        )

    def _generate_programming_exercises(
        self,
        concept: Concept,
        question_type: QuestionType,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate programming exercises"""
        return self._generate_coding_problem(concept, difficulty)

    def _generate_coding_problem(
        self,
        concept: Concept,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate coding problems"""
        problems = {
            DifficultyLevel.BEGINNER: {
                "question": "Write a function that returns the sum of two numbers.",
                "answer": "def sum(a, b):\n    return a + b",
                "test_cases": "sum(2, 3) == 5, sum(0, 0) == 0"
            },
            DifficultyLevel.INTERMEDIATE: {
                "question": "Write a function that returns the factorial of a number.",
                "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "test_cases": "factorial(5) == 120, factorial(0) == 1"
            },
            DifficultyLevel.ADVANCED: {
                "question": "Write a function that checks if a string is a palindrome.",
                "answer": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                "test_cases": "is_palindrome('racecar') == True"
            }
        }

        problem = problems.get(difficulty, problems[DifficultyLevel.BEGINNER])

        return Exercise(
            concept_id=concept.id,
            question_type=QuestionType.CODING,
            difficulty=difficulty,
            question=problem["question"],
            options=[],
            correct_answer=problem["answer"],
            explanation=f"Sample solution:\n{problem['answer']}\n\nTest cases: {problem['test_cases']}",
            hints=[
                "Think about the basic algorithm first",
                "Consider edge cases",
                "Test with simple examples"
            ],
            estimated_time=300 + difficulty.value * 180,
            points=25 * difficulty.value
        )

    def _generate_science_exercises(
        self,
        concept: Concept,
        question_type: QuestionType,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate science exercises"""
        return self._generate_multiple_choice(concept, difficulty)

    def _generate_language_exercises(
        self,
        concept: Concept,
        question_type: QuestionType,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate language exercises"""
        if question_type == QuestionType.FILL_IN_BLANK:
            return self._generate_fill_in_blank(concept, difficulty)
        return self._generate_multiple_choice(concept, difficulty)

    def _generate_history_exercises(
        self,
        concept: Concept,
        question_type: QuestionType,
        difficulty: DifficultyLevel
    ) -> Exercise:
        """Generate history exercises"""
        return self._generate_multiple_choice(concept, difficulty)
