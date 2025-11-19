"""
AI Tutoring Platform - Main Application
Interactive learning application with Windows 11 support
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from typing import Optional
from datetime import datetime

from src.tutoring.tutoring_engine import TutoringEngine
from src.tutoring.models import Student, Exercise, DifficultyLevel


class TutoringApp:
    """
    Main GUI application for the AI Tutoring Platform
    Windows 11 compatible with modern styling
    """

    def __init__(self, root):
        """
        Initialize the application

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("AI Tutoring Platform")
        self.root.geometry("900x700")

        # Windows 11 styling
        self.setup_styling()

        # Initialize tutoring engine
        self.engine = TutoringEngine()

        # Current state
        self.current_student: Optional[Student] = None
        self.current_exercise: Optional[Exercise] = None
        self.session_start_time: Optional[datetime] = None
        self.exercise_start_time: Optional[datetime] = None

        # Build UI
        self.build_ui()

    def setup_styling(self):
        """Setup Windows 11 style theme"""
        style = ttk.Style()

        # Try to use Windows 11 theme
        available_themes = style.theme_names()
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'clam' in available_themes:
            style.theme_use('clam')

        # Custom colors (Windows 11 inspired)
        bg_color = "#F3F3F3"
        accent_color = "#0078D4"
        text_color = "#1F1F1F"

        self.root.configure(bg=bg_color)

        # Configure styles
        style.configure('Title.TLabel',
                       font=('Segoe UI', 24, 'bold'),
                       foreground=text_color,
                       background=bg_color)

        style.configure('Header.TLabel',
                       font=('Segoe UI', 16, 'bold'),
                       foreground=text_color,
                       background=bg_color)

        style.configure('Normal.TLabel',
                       font=('Segoe UI', 11),
                       foreground=text_color,
                       background=bg_color)

        style.configure('Accent.TButton',
                       font=('Segoe UI', 11),
                       foreground='white',
                       background=accent_color)

    def build_ui(self):
        """Build the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame,
            text="🎓 AI Tutoring Platform",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Student section
        self.build_student_section(main_frame, row=1)

        # Subject selection
        self.build_subject_section(main_frame, row=2)

        # Exercise section
        self.build_exercise_section(main_frame, row=3)

        # Progress section
        self.build_progress_section(main_frame, row=4)

        # Buttons
        self.build_buttons_section(main_frame, row=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def build_student_section(self, parent, row):
        """Build student login/selection section"""
        frame = ttk.LabelFrame(parent, text="Student Login", padding="10")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Email entry
        ttk.Label(frame, text="Email:", style='Normal.TLabel').grid(
            row=0, column=0, sticky=tk.W, padx=5
        )
        self.email_entry = ttk.Entry(frame, width=30)
        self.email_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        # Name entry (for new students)
        ttk.Label(frame, text="Name:", style='Normal.TLabel').grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        self.name_entry = ttk.Entry(frame, width=30)
        self.name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)

        # Login button
        self.login_btn = ttk.Button(
            frame,
            text="Login / Register",
            command=self.login_student
        )
        self.login_btn.grid(row=0, column=2, rowspan=2, padx=5)

        # Current student label
        self.student_label = ttk.Label(
            frame,
            text="Not logged in",
            style='Normal.TLabel'
        )
        self.student_label.grid(row=2, column=0, columnspan=3, pady=5)

        frame.columnconfigure(1, weight=1)

    def build_subject_section(self, parent, row):
        """Build subject selection section"""
        frame = ttk.LabelFrame(parent, text="Select Subject", padding="10")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(frame, text="Subject:", style='Normal.TLabel').grid(
            row=0, column=0, sticky=tk.W, padx=5
        )

        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(
            frame,
            textvariable=self.subject_var,
            values=["Mathematics", "Science", "Programming", "Language", "History"],
            state="readonly",
            width=28
        )
        self.subject_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.subject_combo.set("Mathematics")

        self.start_session_btn = ttk.Button(
            frame,
            text="Start Learning Session",
            command=self.start_session,
            state=tk.DISABLED
        )
        self.start_session_btn.grid(row=0, column=2, padx=5)

        frame.columnconfigure(1, weight=1)

    def build_exercise_section(self, parent, row):
        """Build exercise display section"""
        frame = ttk.LabelFrame(parent, text="Current Exercise", padding="10")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        # Question display
        self.question_text = scrolledtext.ScrolledText(
            frame,
            height=8,
            width=80,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            state=tk.DISABLED
        )
        self.question_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Answer entry
        ttk.Label(frame, text="Your Answer:", style='Normal.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.answer_entry = ttk.Entry(frame, width=60)
        self.answer_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Submit button
        self.submit_btn = ttk.Button(
            frame,
            text="Submit Answer",
            command=self.submit_answer,
            state=tk.DISABLED
        )
        self.submit_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # Feedback display
        self.feedback_text = scrolledtext.ScrolledText(
            frame,
            height=4,
            width=80,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            state=tk.DISABLED
        )
        self.feedback_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

    def build_progress_section(self, parent, row):
        """Build progress display section"""
        frame = ttk.LabelFrame(parent, text="Your Progress", padding="10")
        frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.progress_label = ttk.Label(
            frame,
            text="Login to see your progress",
            style='Normal.TLabel'
        )
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        frame.columnconfigure(0, weight=1)

    def build_buttons_section(self, parent, row):
        """Build action buttons section"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, pady=10)

        self.next_exercise_btn = ttk.Button(
            frame,
            text="Next Exercise",
            command=self.load_next_exercise,
            state=tk.DISABLED
        )
        self.next_exercise_btn.grid(row=0, column=0, padx=5)

        self.end_session_btn = ttk.Button(
            frame,
            text="End Session",
            command=self.end_session,
            state=tk.DISABLED
        )
        self.end_session_btn.grid(row=0, column=1, padx=5)

        self.dashboard_btn = ttk.Button(
            frame,
            text="Open Dashboard",
            command=self.open_dashboard
        )
        self.dashboard_btn.grid(row=0, column=2, padx=5)

    # Event handlers

    def login_student(self):
        """Login or register a student"""
        email = self.email_entry.get().strip()
        name = self.name_entry.get().strip()

        if not email:
            messagebox.showerror("Error", "Please enter an email address")
            return

        # Try to get existing student
        student = self.engine.get_student_by_email(email)

        if student:
            self.current_student = student
            messagebox.showinfo("Success", f"Welcome back, {student.name}!")
        else:
            # Create new student
            if not name:
                messagebox.showerror("Error", "Please enter your name for registration")
                return

            student = self.engine.create_student(name, email)
            self.current_student = student
            messagebox.showinfo("Success", f"Welcome, {name}! Your account has been created.")

        # Update UI
        self.student_label.config(text=f"Logged in as: {self.current_student.name}")
        self.start_session_btn.config(state=tk.NORMAL)
        self.update_progress_display()

    def start_session(self):
        """Start a learning session"""
        if not self.current_student:
            messagebox.showerror("Error", "Please login first")
            return

        subject = self.subject_var.get()
        if not subject:
            messagebox.showerror("Error", "Please select a subject")
            return

        # Initialize sample concepts if needed
        concepts = self.engine.get_concepts_by_subject(subject)
        if not concepts:
            self.initialize_sample_concepts(subject)

        # Start session
        session = self.engine.start_learning_session(self.current_student.id, subject)
        self.session_start_time = datetime.now()

        # Update UI
        self.start_session_btn.config(state=tk.DISABLED)
        self.end_session_btn.config(state=tk.NORMAL)
        self.next_exercise_btn.config(state=tk.NORMAL)

        messagebox.showinfo("Session Started", f"Good luck with {subject}!")

        # Load first exercise
        self.load_next_exercise()

    def load_next_exercise(self):
        """Load the next exercise"""
        if not self.current_student:
            return

        subject = self.subject_var.get()
        exercise = self.engine.get_next_exercise(self.current_student.id, subject)

        if not exercise:
            messagebox.showinfo("No Exercises", "No more exercises available. Creating new ones...")
            # Generate exercises for available concepts
            concepts = self.engine.get_concepts_by_subject(subject)
            if concepts:
                self.engine.generate_exercises_for_concept(concepts[0].id, count=5)
                exercise = self.engine.get_next_exercise(self.current_student.id, subject)

        if exercise:
            self.current_exercise = exercise
            self.exercise_start_time = datetime.now()
            self.display_exercise(exercise)
            self.submit_btn.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("No Exercises", "No exercises available at this time")

    def display_exercise(self, exercise: Exercise):
        """Display an exercise"""
        self.question_text.config(state=tk.NORMAL)
        self.question_text.delete(1.0, tk.END)

        # Display question
        self.question_text.insert(tk.END, f"Question ({exercise.difficulty.name}):\n\n")
        self.question_text.insert(tk.END, exercise.question + "\n\n")

        # Display options if multiple choice
        if exercise.options:
            self.question_text.insert(tk.END, "Options:\n")
            for i, option in enumerate(exercise.options, 1):
                self.question_text.insert(tk.END, f"{i}. {option}\n")

        self.question_text.config(state=tk.DISABLED)

        # Clear previous answer and feedback
        self.answer_entry.delete(0, tk.END)
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.config(state=tk.DISABLED)

    def submit_answer(self):
        """Submit the current answer"""
        if not self.current_exercise or not self.current_student:
            return

        answer = self.answer_entry.get().strip()
        if not answer:
            messagebox.showerror("Error", "Please enter an answer")
            return

        # Calculate time spent
        time_spent = int((datetime.now() - self.exercise_start_time).total_seconds())

        # Submit to engine
        result = self.engine.submit_answer(
            self.current_student.id,
            self.current_exercise.id,
            answer,
            time_spent
        )

        # Display feedback
        self.display_feedback(result)

        # Update progress
        self.update_progress_display()

        # Disable submit, enable next
        self.submit_btn.config(state=tk.DISABLED)

    def display_feedback(self, result: dict):
        """Display feedback after answer submission"""
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)

        if result['correct']:
            self.feedback_text.insert(tk.END, "✅ Correct! Well done!\n\n", 'correct')
        else:
            self.feedback_text.insert(tk.END, "❌ Incorrect\n\n", 'incorrect')
            if result.get('correct_answer'):
                self.feedback_text.insert(tk.END, f"Correct answer: {result['correct_answer']}\n\n")

        self.feedback_text.insert(tk.END, f"Explanation: {result['explanation']}\n\n")
        self.feedback_text.insert(
            tk.END,
            f"Mastery: {result['mastery_probability']*100:.1f}% ({result['mastery_level']})\n"
        )

        if result.get('points_earned'):
            self.feedback_text.insert(tk.END, f"Points earned: +{result['points_earned']}\n")

        if result.get('new_achievements'):
            self.feedback_text.insert(tk.END, "\n🏆 New Achievements Unlocked!\n")

        self.feedback_text.config(state=tk.DISABLED)

    def update_progress_display(self):
        """Update progress display"""
        if not self.current_student:
            return

        summary = self.engine.get_performance_summary(self.current_student.id)

        progress_text = (
            f"📊 Exercises: {summary['total_exercises_completed']} | "
            f"✅ Mastered: {summary['mastered_concepts']}/{summary['total_concepts']} | "
            f"🎯 Accuracy: {summary['accuracy']*100:.1f}% | "
            f"🔥 Streak: {summary['streak_days']} days"
        )

        self.progress_label.config(text=progress_text)

    def end_session(self):
        """End the current learning session"""
        if self.engine.current_session:
            session = self.engine.end_learning_session()

            if session.exercises_completed > 0:
                accuracy = session.correct_answers / session.exercises_completed
                messagebox.showinfo(
                    "Session Complete",
                    f"Great work!\n\n"
                    f"Exercises completed: {session.exercises_completed}\n"
                    f"Accuracy: {accuracy*100:.1f}%\n"
                    f"Time spent: {session.total_time // 60} minutes\n"
                    f"Performance score: {session.performance_score:.1f}"
                )

        # Reset UI
        self.start_session_btn.config(state=tk.NORMAL)
        self.end_session_btn.config(state=tk.DISABLED)
        self.next_exercise_btn.config(state=tk.DISABLED)
        self.submit_btn.config(state=tk.DISABLED)

        self.question_text.config(state=tk.NORMAL)
        self.question_text.delete(1.0, tk.END)
        self.question_text.config(state=tk.DISABLED)

    def open_dashboard(self):
        """Open the web dashboard in a new thread"""
        def run_dashboard():
            from src.tutoring.dashboard import TutoringDashboard
            dashboard = TutoringDashboard(self.engine.database)
            dashboard.run(debug=False, port=8050)

        thread = threading.Thread(target=run_dashboard, daemon=True)
        thread.start()

        messagebox.showinfo(
            "Dashboard",
            "Opening dashboard in web browser...\n\n"
            "URL: http://127.0.0.1:8050"
        )

    def initialize_sample_concepts(self, subject: str):
        """Initialize sample concepts for a subject"""
        concepts_data = {
            "Mathematics": [
                ("Addition", "Adding two or more numbers together", DifficultyLevel.BEGINNER),
                ("Subtraction", "Taking one number away from another", DifficultyLevel.BEGINNER),
                ("Multiplication", "Repeated addition of the same number", DifficultyLevel.INTERMEDIATE),
                ("Division", "Splitting a number into equal parts", DifficultyLevel.INTERMEDIATE),
                ("Algebra", "Solving equations with variables", DifficultyLevel.ADVANCED),
            ],
            "Programming": [
                ("Variables", "Storing data in memory", DifficultyLevel.BEGINNER),
                ("Functions", "Reusable blocks of code", DifficultyLevel.INTERMEDIATE),
                ("Loops", "Repeating code multiple times", DifficultyLevel.INTERMEDIATE),
                ("Recursion", "Functions that call themselves", DifficultyLevel.ADVANCED),
            ]
        }

        concepts = concepts_data.get(subject, [])
        for name, description, difficulty in concepts:
            self.engine.create_concept(name, description, subject, difficulty)

    def on_close(self):
        """Handle window close"""
        if self.engine.current_session:
            if messagebox.askyesno("Confirm", "End current session and exit?"):
                self.engine.close()
                self.root.destroy()
        else:
            self.engine.close()
            self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = TutoringApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
