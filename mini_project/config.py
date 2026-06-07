import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask session secret; override with environment variable in production
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret')

    # Use SQLite for development if MySQL is not available
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        None  # Will be set in app.py if MySQL fails
    )
    
    if DATABASE_URL:
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
    else:
        # Default to SQLite for development
        SQLALCHEMY_DATABASE_URI = 'sqlite:///student_management.db'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True

    # Gmail SMTP settings for OTP delivery
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER', os.getenv('MAIL_USERNAME', ''))
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://127.0.0.1:5000')