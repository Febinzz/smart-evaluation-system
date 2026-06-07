from datetime import datetime
from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash


class Student(db.Model):
    __tablename__ = 'student'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    username = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    otp_code = db.Column(db.String(6), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)
    # admission / roll number – shown as "Admission No" on dashboard
    roll_number = db.Column(db.String(50), unique=True)
    semester = db.Column(db.String(20))
    department = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    photo = db.Column(db.Text)  # base64 or URL
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password, raw_password)


class Teacher(db.Model):
    __tablename__ = 'teacher'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    username = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    otp_code = db.Column(db.String(6), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)
    department = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    photo = db.Column(db.Text)  # base64 or URL
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password, raw_password)


class Exam(db.Model):
    __tablename__ = 'exam'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Integer)  # in minutes
    questions_count = db.Column(db.Integer)
    created_by = db.Column(db.Integer, db.ForeignKey('teacher.id'))
    status_active = db.Column(db.Boolean, default=False)


class ExamStudent(db.Model):
    __tablename__ = 'exam_student'

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'))
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    status = db.Column(db.String(50), default='not-attempted')  # in-progress, completed
    score = db.Column(db.Integer, default=0)
    answers = db.Column(db.JSON)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)


class Question(db.Model):
    __tablename__ = 'question'

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'))
    question_text = db.Column(db.Text)
    correct_answer = db.Column(db.Text)
    question_type = db.Column(db.String(30), default='direct')
