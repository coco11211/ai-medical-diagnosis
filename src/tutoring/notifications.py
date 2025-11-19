"""
Windows 11 Notifications System
Desktop notifications for the tutoring platform
"""
import platform
from datetime import datetime
from typing import Optional


class NotificationManager:
    """
    Cross-platform notification manager with Windows 11 support
    """

    def __init__(self):
        """Initialize notification manager"""
        self.system = platform.system()
        self._init_platform_notifier()

    def _init_platform_notifier(self):
        """Initialize platform-specific notifier"""
        if self.system == "Windows":
            try:
                from win10toast import ToastNotifier
                self.toaster = ToastNotifier()
                self.windows_available = True
            except ImportError:
                print("Warning: win10toast not available. Installing fallback notification system.")
                self.windows_available = False
        else:
            self.windows_available = False

    def send_notification(
        self,
        title: str,
        message: str,
        icon: Optional[str] = None,
        duration: int = 5
    ) -> bool:
        """
        Send a desktop notification

        Args:
            title: Notification title
            message: Notification message
            icon: Optional icon path
            duration: Duration in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system == "Windows" and self.windows_available:
                # Use Windows 11 native notifications
                self.toaster.show_toast(
                    title=title,
                    msg=message,
                    icon_path=icon,
                    duration=duration,
                    threaded=True
                )
                return True
            else:
                # Fallback to console output
                self._console_notification(title, message)
                return True
        except Exception as e:
            print(f"Notification error: {e}")
            self._console_notification(title, message)
            return False

    def _console_notification(self, title: str, message: str):
        """Fallback console notification"""
        print("\n" + "="*50)
        print(f"📢 {title}")
        print("-"*50)
        print(f"{message}")
        print("="*50 + "\n")

    # Pre-defined notification types

    def notify_session_start(self, student_name: str, subject: str):
        """Notify when learning session starts"""
        self.send_notification(
            title="Learning Session Started",
            message=f"{student_name} started learning {subject}",
            duration=3
        )

    def notify_session_end(
        self,
        student_name: str,
        exercises_completed: int,
        accuracy: float,
        time_spent: int
    ):
        """Notify when learning session ends"""
        minutes = time_spent // 60
        message = (
            f"{student_name} completed {exercises_completed} exercises\n"
            f"Accuracy: {accuracy*100:.1f}% | Time: {minutes} minutes"
        )
        self.send_notification(
            title="Session Complete! 🎉",
            message=message,
            duration=7
        )

    def notify_achievement(self, student_name: str, achievement: str):
        """Notify about new achievement"""
        self.send_notification(
            title="New Achievement Unlocked! 🏆",
            message=f"{student_name}: {achievement}",
            duration=5
        )

    def notify_mastery(self, student_name: str, concept: str):
        """Notify when concept is mastered"""
        self.send_notification(
            title="Concept Mastered! ⭐",
            message=f"{student_name} has mastered {concept}!",
            duration=5
        )

    def notify_review_due(self, student_name: str, concepts_count: int):
        """Notify about pending reviews"""
        self.send_notification(
            title="Time for Review! 📚",
            message=f"{student_name}, you have {concepts_count} concepts ready for review",
            duration=5
        )

    def notify_streak(self, student_name: str, days: int):
        """Notify about study streak"""
        self.send_notification(
            title=f"{days}-Day Streak! 🔥",
            message=f"Congratulations {student_name}! Keep it going!",
            duration=5
        )

    def notify_goal_reached(self, student_name: str, goal: str):
        """Notify when goal is reached"""
        self.send_notification(
            title="Goal Reached! 🎯",
            message=f"{student_name} achieved: {goal}",
            duration=7
        )

    def notify_encouragement(self, student_name: str):
        """Send encouraging notification"""
        messages = [
            "You're making great progress!",
            "Keep up the excellent work!",
            "Learning is a journey, enjoy the process!",
            "Every practice session counts!",
            "You're getting stronger every day!"
        ]
        import random
        message = random.choice(messages)

        self.send_notification(
            title="Keep Going! 💪",
            message=f"{student_name}, {message}",
            duration=4
        )


class AchievementSystem:
    """
    Manages student achievements and badges
    """

    def __init__(self, notification_manager: NotificationManager):
        """
        Initialize achievement system

        Args:
            notification_manager: NotificationManager instance
        """
        self.notifier = notification_manager
        self.achievements = self._define_achievements()

    def _define_achievements(self):
        """Define all available achievements"""
        return {
            # Exercise milestones
            "first_exercise": {
                "name": "Getting Started",
                "description": "Completed first exercise",
                "icon": "🌟"
            },
            "10_exercises": {
                "name": "Dedicated Learner",
                "description": "Completed 10 exercises",
                "icon": "📖"
            },
            "50_exercises": {
                "name": "Practice Makes Perfect",
                "description": "Completed 50 exercises",
                "icon": "💯"
            },
            "100_exercises": {
                "name": "Century Club",
                "description": "Completed 100 exercises",
                "icon": "🎯"
            },

            # Mastery milestones
            "first_mastery": {
                "name": "First Victory",
                "description": "Mastered first concept",
                "icon": "⭐"
            },
            "5_masteries": {
                "name": "Knowledge Builder",
                "description": "Mastered 5 concepts",
                "icon": "🏗️"
            },
            "10_masteries": {
                "name": "Expert Scholar",
                "description": "Mastered 10 concepts",
                "icon": "🎓"
            },

            # Streak milestones
            "3_day_streak": {
                "name": "On Fire",
                "description": "3-day study streak",
                "icon": "🔥"
            },
            "7_day_streak": {
                "name": "Week Warrior",
                "description": "7-day study streak",
                "icon": "💪"
            },
            "30_day_streak": {
                "name": "Unstoppable",
                "description": "30-day study streak",
                "icon": "🚀"
            },

            # Accuracy milestones
            "perfect_10": {
                "name": "Perfect Ten",
                "description": "10 correct answers in a row",
                "icon": "✨"
            },
            "accuracy_90": {
                "name": "Precision Master",
                "description": "Maintained 90% accuracy over 20 exercises",
                "icon": "🎯"
            },

            # Time milestones
            "5_hours": {
                "name": "Dedicated Student",
                "description": "Spent 5 hours learning",
                "icon": "⏰"
            },
            "20_hours": {
                "name": "Time Investment",
                "description": "Spent 20 hours learning",
                "icon": "📚"
            },

            # Subject mastery
            "subject_master": {
                "name": "Subject Master",
                "description": "Mastered all concepts in a subject",
                "icon": "👑"
            }
        }

    def check_achievements(
        self,
        student,
        new_attempt=None
    ) -> list:
        """
        Check if student has earned any new achievements

        Args:
            student: Student object
            new_attempt: Optional new Attempt object

        Returns:
            List of newly earned achievement IDs
        """
        new_achievements = []

        # Exercise count achievements
        if student.total_exercises_completed == 1:
            new_achievements.append("first_exercise")
        elif student.total_exercises_completed == 10:
            new_achievements.append("10_exercises")
        elif student.total_exercises_completed == 50:
            new_achievements.append("50_exercises")
        elif student.total_exercises_completed == 100:
            new_achievements.append("100_exercises")

        # Mastery achievements
        mastered_count = sum(
            1 for state in student.knowledge_states.values()
            if state.mastery_probability >= 0.95
        )

        if mastered_count == 1 and "first_mastery" not in student.achievements:
            new_achievements.append("first_mastery")
        elif mastered_count == 5 and "5_masteries" not in student.achievements:
            new_achievements.append("5_masteries")
        elif mastered_count == 10 and "10_masteries" not in student.achievements:
            new_achievements.append("10_masteries")

        # Streak achievements
        if student.streak_days == 3 and "3_day_streak" not in student.achievements:
            new_achievements.append("3_day_streak")
        elif student.streak_days == 7 and "7_day_streak" not in student.achievements:
            new_achievements.append("7_day_streak")
        elif student.streak_days == 30 and "30_day_streak" not in student.achievements:
            new_achievements.append("30_day_streak")

        # Time achievements
        hours = student.total_time_spent / 3600
        if hours >= 5 and "5_hours" not in student.achievements:
            new_achievements.append("5_hours")
        elif hours >= 20 and "20_hours" not in student.achievements:
            new_achievements.append("20_hours")

        # Add to student's achievements and notify
        for achievement_id in new_achievements:
            if achievement_id not in student.achievements:
                student.achievements.append(achievement_id)
                achievement = self.achievements[achievement_id]
                self.notifier.notify_achievement(
                    student.name,
                    f"{achievement['icon']} {achievement['name']}: {achievement['description']}"
                )

        return new_achievements
