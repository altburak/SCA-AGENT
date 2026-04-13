"""Event handler for application lifecycle."""


class EventHandler:
    """Handles lifecycle events."""

    def __init__(self, name: str):
        self.name = name
        self.state = "idle"

    def on_start(self):
        self.state = "running"
        print(f"{self.name}: started")

    def on_pause(self):
        if self.state == "running":
            self.state = "paused"
            print(f"{self.name}: paused")

    def on_resume(self):
        if self.state == "paused":
            self.state = "running"
            print(f"{self.name}: resumed")

    def on_stop(self):
        self.state = "stopped"
        print(f"{self.name}: stopped")