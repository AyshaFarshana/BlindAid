# mode_manager.py

class ModeManager:
    def __init__(self):
        self.current_mode = "road"  # default mode

    def switch_mode(self, new_mode):
        if new_mode in ["road", "market"]:
            self.current_mode = new_mode
            return True
        return False

    def get_mode(self):
        return self.current_mode