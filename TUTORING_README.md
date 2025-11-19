](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Windows 11 Compatibility](#windows-11-compatibility)

## Features

### 🧠 Adaptive Learning Engine
- **FSRS Algorithm**: Advanced spaced repetition for optimal review scheduling
- **Personalized Pacing**: Adapts difficulty based on student performance
- **Intelligent Scheduling**: Automatically determines when to review concepts
- **Learning Optimization**: Maximizes retention while minimizing study time

### 📊 Knowledge Tracing
- **Bayesian Knowledge Tracing (BKT)**: Tracks probability of concept mastery
- **Real-time Updates**: Adjusts knowledge estimates after each exercise
- **Mastery Levels**: Five levels from "Not Learned" to "Mastered"
- **Predictive Analytics**: Forecasts student performance on upcoming exercises

### 📝 Exercise Generation
- **Multiple Question Types**:
  - Multiple Choice
  - True/False
  - Short Answer
  - Fill in the Blank
  - Math Problems
  - Coding Challenges
  - Essay Questions

- **Subject Coverage**:
  - Mathematics (Algebra, Geometry, Calculus)
  - Science
  - Programming (Python, algorithms)
  - Language Arts
  - History

- **Difficulty Scaling**: Automatically adjusts from Beginner to Expert levels
- **Hint System**: Progressive hints to guide learning
- **Detailed Explanations**: Comprehensive feedback for every answer

### 📈 Progress Tracking
- **Comprehensive Analytics**:
  - Mastery percentage by subject
  - Accuracy rates
  - Learning curves
  - Time distribution
  - Streak tracking

- **Performance Reports**:
  - Overall summaries
  - Subject breakdowns
  - Weak areas identification
  - Strength analysis
  - Personalized recommendations

### 🖥️ Modern UI
- **Windows 11 Native Styling**: Matches Windows 11 design language
- **Interactive Dashboard**: Web-based analytics with Plotly/Dash
- **Desktop Application**: Tkinter-based GUI with modern aesthetics
- **Responsive Design**: Adapts to different screen sizes

### 🔔 Notifications (Windows 11)
- **Native Toast Notifications**: Uses Windows 11 notification system
- **Achievement Alerts**: Celebrates milestones and accomplishments
- **Session Summaries**: End-of-session performance reports
- **Review Reminders**: Notifies when concepts need review

### 🏆 Achievement System
- **Milestone Badges**: Unlock achievements for progress
- **Streak Rewards**: Celebrate consecutive study days
- **Mastery Awards**: Recognition for concept mastery
- **Motivational Feedback**: Encouragement and positive reinforcement

## Installation

### Prerequisites
- **Windows 11** (also compatible with Windows 10, Linux, macOS)
- **Python 3.8 or higher**
- **4GB RAM** minimum (8GB recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ai-medical-diagnosis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For Windows 11 notifications, ensure `win10toast` is installed:
```bash
pip install win10toast
```

### Step 3: Verify Installation

```bash
python tutoring_app.py
```

## Quick Start

### Launch the Desktop Application

```bash
python tutoring_app.py
```

This opens the main GUI where you can:
1. **Login/Register** as a student
2. **Select a subject** to study
3. **Start a learning session**
4. **Complete exercises** with real-time feedback
5. **Track your progress** with detailed analytics

### Launch the Dashboard

```bash
python -c "from src.tutoring.dashboard import TutoringDashboard; from src.tutoring.database import DatabaseManager; dashboard = TutoringDashboard(DatabaseManager()); dashboard.run()"
```

Or click "Open Dashboard" in the main application.

The dashboard will be available at: **http://127.0.0.1:8050**

### Python API Usage

```python
from src.tutoring.tutoring_engine import TutoringEngine

# Initialize the engine
engine = TutoringEngine()

# Create a student
student = engine.create_student(
    name="Alice Johnson",
    email="alice@example.com",
    grade_level=10
)

# Create some concepts
concept = engine.create_concept(
    name="Quadratic Equations",
    description="Solving equations of the form ax² + bx + c = 0",
    subject="Mathematics",
    difficulty=DifficultyLevel.INTERMEDIATE
)

# Start a learning session
session = engine.start_learning_session(student.id, "Mathematics")

# Get next exercise
exercise = engine.get_next_exercise(student.id, "Mathematics")

# Submit an answer
result = engine.submit_answer(
    student_id=student.id,
    exercise_id=exercise.id,
    answer="x = 2",
    time_spent=45
)

# Check feedback
print(f"Correct: {result['correct']}")
print(f"Mastery: {result['mastery_probability']:.1%}")

# End session
engine.end_learning_session()

# Get progress report
report = engine.get_progress_report(student.id)
```

## Architecture

### Core Components

```
src/tutoring/
├── models.py                 # Data models (Student, Concept, Exercise, etc.)
├── knowledge_tracer.py       # Bayesian Knowledge Tracing
├── adaptive_learning.py      # FSRS spaced repetition algorithm
├── exercise_generator.py     # Dynamic exercise generation
├── progress_tracker.py       # Analytics and progress tracking
├── database.py              # SQLite persistence layer
├── dashboard.py             # Web-based analytics dashboard
├── notifications.py         # Windows 11 notifications
└── tutoring_engine.py       # Main orchestration engine
```

### Algorithms

#### 1. Bayesian Knowledge Tracing (BKT)

Tracks four key parameters:
- **P(L)**: Probability of learning (0.3)
- **P(G)**: Probability of guessing (0.25)
- **P(S)**: Probability of slip/mistake (0.1)
- **P(T)**: Probability of initial knowledge (0.0)

Updates knowledge state using Bayes' theorem after each attempt.

#### 2. FSRS (Free Spaced Repetition Scheduler)

Uses three main variables:
- **Stability (S)**: Memory stability in days
- **Difficulty (D)**: Inherent difficulty (0-1)
- **Retrievability (R)**: Current memory strength (0-1)

Calculates optimal review intervals to maintain 90% retrievability.

#### 3. Adaptive Exercise Selection

Scores concepts based on:
- Low retrievability (needs review)
- Not yet mastered (learning in progress)
- Overdue reviews
- Prerequisites met

## Usage Guide

### For Students

1. **Login/Register**: Enter your email and name (first time only)

2. **Select Subject**: Choose what you want to learn
   - Mathematics
   - Science
   - Programming
   - Language
   - History

3. **Start Session**: Click "Start Learning Session"

4. **Complete Exercises**:
   - Read the question carefully
   - Enter your answer
   - Click "Submit Answer"
   - Review feedback and explanation
   - Click "Next Exercise" to continue

5. **Track Progress**:
   - View real-time stats at the bottom
   - Open dashboard for detailed analytics
   - Earn achievements and badges

6. **End Session**: Click "End Session" when done

### For Tutors/Teachers

1. **View Student Progress**:
   - Open the dashboard
   - Select a student
   - Review analytics and weak areas

2. **Create Custom Concepts**:
   ```python
   engine.create_concept(
       name="Photosynthesis",
       description="Process by which plants convert light to energy",
       subject="Science",
       difficulty=DifficultyLevel.INTERMEDIATE
   )
   ```

3. **Generate Exercises**:
   ```python
   engine.generate_exercises_for_concept(
       concept_id=concept.id,
       count=10,
       question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER]
   )
   ```

## Dashboard

The web dashboard provides comprehensive analytics:

### Key Metrics Cards
- Total concepts studied
- Concepts mastered
- Overall accuracy
- Study streak

### Visualizations
- **Mastery by Subject**: Bar chart showing mastery percentage per subject
- **Progress Over Time**: Line chart tracking performance across sessions
- **Time Distribution**: Pie chart of time spent per subject
- **Mastery Levels**: Distribution of concepts across mastery levels

### Insights
- **Weak Areas**: Concepts that need more practice
- **Strengths**: Well-mastered concepts
- **Recommendations**: Personalized study suggestions

### Accessing the Dashboard

**Method 1** (From GUI): Click "Open Dashboard" button

**Method 2** (Command Line):
```bash
python -c "from src.tutoring.dashboard import TutoringDashboard; from src.tutoring.database import DatabaseManager; TutoringDashboard(DatabaseManager()).run()"
```

**Method 3** (Custom Script):
```python
from src.tutoring.dashboard import TutoringDashboard
from src.tutoring.database import DatabaseManager

db = DatabaseManager("tutoring_data.db")
dashboard = TutoringDashboard(db)
dashboard.run(debug=False, port=8050)
```

## Configuration

The system uses SQLite for data persistence. The database is automatically created at:
```
tutoring_data.db
```

### Customizing BKT Parameters

```python
from src.tutoring.knowledge_tracer import KnowledgeTracer

tracer = KnowledgeTracer(
    p_learn=0.3,      # Learning rate
    p_guess=0.25,     # Guess probability
    p_slip=0.1,       # Slip probability
    p_init=0.0,       # Initial knowledge
    mastery_threshold=0.95  # Mastery threshold
)
```

### Customizing FSRS Parameters

```python
from src.tutoring.adaptive_learning import AdaptiveLearningEngine

engine = AdaptiveLearningEngine(
    initial_stability=1.0,              # Days
    initial_difficulty=0.5,             # 0-1
    retrievability_threshold=0.9        # Target retrievability
)
```

## Windows 11 Compatibility

### Native Notifications

The platform uses **win10toast** for native Windows 11 toast notifications:

```python
from src.tutoring.notifications import NotificationManager

notifier = NotificationManager()
notifier.send_notification(
    title="Achievement Unlocked!",
    message="You've mastered 10 concepts!",
    duration=5
)
```

### Windows 11 Styling

The GUI uses Windows 11 design principles:
- **Segoe UI** font family
- Modern color scheme (#F3F3F3 background, #0078D4 accent)
- Rounded corners and shadows
- Smooth animations

### Installation on Windows 11

All dependencies are Windows 11 compatible:
```bash
pip install -r requirements.txt
```

For best results, ensure you have:
- Windows 11 version 22H2 or later
- Python installed from Microsoft Store or python.org
- Windows Terminal (optional, for better console experience)

## Database Schema

### Tables

- **students**: Student profiles and overall stats
- **concepts**: Learning concepts and prerequisites
- **exercises**: Practice exercises and questions
- **attempts**: Student exercise attempts
- **knowledge_states**: BKT state for each student-concept pair
- **learning_sessions**: Learning session records

### Backup

To backup your data:
```bash
cp tutoring_data.db tutoring_data_backup.db
```

## Advanced Features

### Custom Exercise Templates

Create your own exercise generator:

```python
from src.tutoring.exercise_generator import ExerciseGenerator
from src.tutoring.models import Concept, Exercise, QuestionType

generator = ExerciseGenerator()

# Register custom generator
def my_custom_generator(concept, difficulty):
    return Exercise(
        concept_id=concept.id,
        question_type=QuestionType.SHORT_ANSWER,
        question=f"Explain {concept.name}",
        # ... other fields
    )

# Use it
exercise = my_custom_generator(concept, DifficultyLevel.INTERMEDIATE)
```

### Learning Path Creation

```python
from src.tutoring.models import LearningPath

path = LearningPath(
    student_id=student.id,
    subject="Mathematics",
    target_concepts=[concept1.id, concept2.id, concept3.id]
)
```

### Export Progress Data

```python
import json

report = engine.get_progress_report(student.id)

with open('progress_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
```

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: Notifications not showing on Windows 11

**Solution**:
1. Check Windows notification settings
2. Ensure win10toast is installed: `pip install win10toast`
3. Enable notifications for Python in Windows settings

### Issue: Dashboard won't open

**Solution**:
1. Check port 8050 is not in use
2. Try a different port: `dashboard.run(port=8051)`
3. Check firewall settings

### Issue: Database locked

**Solution**:
- Close all connections to the database
- Restart the application
- If persists, backup and delete `tutoring_data.db`

## Performance Tips

1. **Generate Exercises in Batches**: Pre-generate exercises for better performance
   ```python
   engine.generate_exercises_for_concept(concept.id, count=50)
   ```

2. **Optimize Database**: SQLite auto-optimizes, but you can vacuum:
   ```python
   engine.database.conn.execute("VACUUM")
   ```

3. **Limit History**: Keep only recent sessions for faster queries

## Roadmap

### Planned Features
- [ ] AI-powered exercise generation using GPT models
- [ ] Voice input for answers
- [ ] Mobile app (iOS/Android)
- [ ] Multiplayer learning challenges
- [ ] Teacher dashboard with class management
- [ ] Integration with popular LMS platforms
- [ ] Video lessons and tutorials
- [ ] Gamification with leaderboards
- [ ] Export to PDF reports
- [ ] Cloud sync across devices

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational purposes.

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## Acknowledgments

Built with:
- **Python**: Core language
- **NumPy/Pandas**: Data processing
- **Plotly/Dash**: Interactive visualizations
- **Tkinter**: Desktop GUI
- **SQLite**: Data persistence
- **Scikit-learn**: Machine learning utilities

**Algorithms**:
- Bayesian Knowledge Tracing (BKT)
- FSRS (Free Spaced Repetition Scheduler)

---

**Happy Learning! 📚🎓**

Transform your education with personalized, adaptive AI tutoring.
