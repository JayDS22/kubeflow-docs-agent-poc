"""
Unit tests for the keyword-based intent router.

Covers docs, issues, greeting, and out-of-scope classification.
"""
from agent.router import classify_intent


class TestDocsIntent:
    def test_install_query(self):
        assert classify_intent("How do I install Kubeflow?") == "docs"

    def test_pipeline_query(self):
        assert classify_intent("How to create a Kubeflow Pipeline?") == "docs"

    def test_kserve_query(self):
        assert classify_intent("What is KServe?") == "docs"

    def test_notebook_query(self):
        assert classify_intent("How do I configure notebooks in Kubeflow?") == "docs"

    def test_katib_query(self):
        assert classify_intent("How to use Katib for hyperparameter tuning?") == "docs"

    def test_getting_started(self):
        assert classify_intent("Getting started with Kubeflow") == "docs"


class TestIssuesIntent:
    def test_error_query(self):
        assert classify_intent("I'm getting an error with my pipeline") == "issues"

    def test_crash_query(self):
        assert classify_intent("Pod keeps crashing with OOM") == "issues"

    def test_permission_denied(self):
        assert classify_intent("Permission denied 403 when accessing dashboard") == "issues"

    def test_debug_query(self):
        assert classify_intent("How to debug CrashLoopBackOff") == "issues"

    def test_not_working(self):
        assert classify_intent("Pipeline is not working") == "issues"


class TestGreetingIntent:
    def test_hello(self):
        assert classify_intent("hello") == "greeting"

    def test_hi(self):
        assert classify_intent("hi") == "greeting"

    def test_hey(self):
        assert classify_intent("hey") == "greeting"

    def test_good_morning(self):
        assert classify_intent("good morning") == "greeting"


class TestOutOfScope:
    def test_weather(self):
        assert classify_intent("What's the weather today?") == "out_of_scope"

    def test_sports(self):
        assert classify_intent("Who won the Super Bowl?") == "out_of_scope"

    def test_random(self):
        assert classify_intent("Tell me a joke about cats") == "out_of_scope"
