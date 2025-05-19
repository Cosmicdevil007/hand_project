class Login:
    def __init__(self, user_db):
        self.user_db = user_db  # A dictionary to store user credentials

    def validate_credentials(self, username, password):
        """Validate user credentials against the stored user database."""
        if username in self.user_db and self.user_db[username] == password:
            return True
        return False

    def login(self, username, password):
        """Handle user login."""
        if self.validate_credentials(username, password):
            return f"User {username} logged in successfully."
        else:
            return "Invalid username or password."

    def logout(self, username):
        """Handle user logout."""
        return f"User {username} logged out successfully."