import os
import smtplib
import ssl
import requests
from flask import Flask, request, jsonify, session, redirect
from config import Config
from extensions import db
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
import re
import secrets
from email.mime.text import MIMEText
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# make sure models are imported so SQLAlchemy knows about them
import models
from models import Student, Teacher, Exam, ExamStudent, Question

# import models module so that SQLAlchemy is aware of all subclasses of
# db.Model before create_all() is called.  the import has no other side
# effects; it merely triggers the class definitions in models.py.
import models
SKIP_MODEL_LOAD = os.getenv('SKIP_MODEL_LOAD', '').lower() in ('1', 'true', 'yes')
if SKIP_MODEL_LOAD:
    print("SKIP_MODEL_LOAD is set — skipping heavy AI model loading")
    sbert = None
    nli = None
else:
    try:
        print("Loading AI Models...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        nli = pipeline("text-classification", model="roberta-large-mnli")
        print("Models Loaded!")
    except Exception as model_err:
        print(f"Failed to load AI models: {model_err}")
        sbert = None
        nli = None

APP_TIMEZONE_OFFSET_MINUTES = int(os.getenv('APP_TIMEZONE_OFFSET_MINUTES', '330'))
VALID_DEPARTMENTS = {'CSA', 'CSB', 'CYBER', 'AIDS', 'AI', 'MECH', 'CIVIL', 'EEE', 'EC'}


def app_now():
    return datetime.utcnow() + timedelta(minutes=APP_TIMEZONE_OFFSET_MINUTES)


def _is_valid_email(email: str) -> bool:
    if not email:
        return False
    return re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email) is not None


def _is_valid_phone(phone: str) -> bool:
    return re.match(r'^\d{10}$', phone or '') is not None


def normalize(code: str) -> str:
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    return re.sub(r'\s+', '', code)


def evaluate_code_question(question, correct_blank, student_answer):
    student_n = normalize(student_answer)
    answer_n = normalize(correct_blank)

    if not answer_n:
        return student_n == answer_n

    # Find the real blank placeholder.
    # Prefer underscore runs with length >= 2 so identifiers like sum_numbers
    # don't get treated as blanks. If none exist, fall back to the longest run.
    blank_runs = list(re.finditer(r"_+", question or ""))
    if blank_runs:
        preferred = [m for m in blank_runs if len(m.group(0)) >= 2]
        target = max(preferred or blank_runs, key=lambda m: len(m.group(0)))
        prefix_raw = question[:target.start()]
        suffix_raw = question[target.end():]
    else:
        prefix_raw = question or ""
        suffix_raw = ""

    prefix_n = normalize(prefix_raw)
    suffix_n = normalize(suffix_raw)

    # The correct blank must appear in student answer.
    # Then trim any overlapping left/right context around each match.
    start = 0
    while True:
        idx = student_n.find(answer_n, start)
        if idx == -1:
            break

        left_side = student_n[:idx]
        right_side = student_n[idx + len(answer_n):]

        # Trim longest suffix of left_side that matches ending part of prefix context
        left_overlap = 0
        if prefix_n and left_side:
            max_len = min(len(prefix_n), len(left_side))
            for length in range(max_len, 0, -1):
                if left_side.endswith(prefix_n[-length:]):
                    left_overlap = length
                    break

        # Trim longest prefix of right_side that matches starting part of suffix context
        right_overlap = 0
        if suffix_n and right_side:
            max_len = min(len(suffix_n), len(right_side))
            for length in range(max_len, 0, -1):
                if right_side.startswith(suffix_n[:length]):
                    right_overlap = length
                    break

        left_remaining = left_side[:-left_overlap] if left_overlap else left_side
        right_remaining = right_side[right_overlap:] if right_overlap else right_side

        # Accept if nothing remains outside the matched answer + trimmed context
        if left_remaining == "" and right_remaining == "":
            return True

        start = idx + 1

    return False


def evaluate_fill_blank(question, teacher_answer, student_answer):
    teacher = re.sub(r'[^a-z0-9]', '', teacher_answer.lower().strip())
    student = re.sub(r'[^a-z0-9]', '', student_answer.lower().strip())

    return teacher == student


def evaluate_semantic(question, teacher_answer, student_answer, sim_threshold=0.65):
    teacher_full = re.sub(r"_+", teacher_answer, question, count=1)
    student_full = re.sub(r"_+", student_answer, question, count=1)
    # If models weren't loaded (dev mode or OOM), fall back to simple comparison
    if nli is None or sbert is None:
        return student_answer.lower().strip() == teacher_answer.lower().strip()

    nli_input = f"{teacher_full} </s></s> {student_full}"
    nli_result = nli(nli_input)[0]

    if nli_result["label"] == "CONTRADICTION" and nli_result["score"] > 0.6:
        return False

    if nli_result["label"] == "ENTAILMENT" and nli_result["score"] > 0.6:
        return True

    emb_teacher = sbert.encode(teacher_full, convert_to_tensor=True)
    emb_student = sbert.encode(student_full, convert_to_tensor=True)

    similarity = util.cos_sim(emb_teacher, emb_student).item()

    return similarity >= sim_threshold


def evaluate_numerical(teacher_answer, student_answer):
    try:
        return round(float(teacher_answer), 1) == round(float(student_answer), 1)
    except:
        return False


def create_app():
    """Application factory that configures Flask and extensions."""
    # serve static assets (css/js/images, and the existing html files) from
    # the project root so that the original index.html can work without any
    # modification to its relative URLs.
    app = Flask(__name__, static_folder='.', static_url_path='')
    app.config.from_object(Config)
    app.logger.info(f"MAIL_USERNAME loaded: {'yes' if app.config.get('MAIL_USERNAME') else 'no'}")
    # set secret key for session
    app.secret_key = app.config.get('SECRET_KEY')
    # strengthen session cookie defaults for safety
    app.config.setdefault('SESSION_COOKIE_HTTPONLY', True)
    app.config.setdefault('SESSION_COOKIE_SAMESITE', 'Lax')

    # initialize extensions with the app
    db.init_app(app)

    # Global error handler to return JSON for API endpoints
    @app.errorhandler(Exception)
    def handle_error(e):
        app.logger.error(f"Unhandled error: {e}")
        if request.path.startswith('/api/'):
            return jsonify(message='server error'), 500
        raise e

    # make sure models are imported after db.bind; this is normally handled
    # by the top-level import above but kept here for clarity.
    # (``models`` already imported at module level.)
    
    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        try:
            data = request.json or {}

            question = data.get("question", "").strip()
            teacher_answer = data.get("teacher_answer", "").strip()
            student_answer = data.get("student_answer", "").strip()
            q_type = data.get("type", "direct")

            print("DEBUG student_answer:", repr(student_answer))

            # 🔥 FINAL EMPTY PROTECTION
            if student_answer == "" or student_answer.lower() == "not answered":
                print("⚠️ Empty answer detected → Marking incorrect")
                return jsonify({"correct": False})

            if q_type == "numerical":
                result = evaluate_numerical(teacher_answer, student_answer)

            elif q_type == "code":
                result = evaluate_code_question(question, teacher_answer, student_answer)

            elif q_type == "direct":
                result = evaluate_fill_blank(question, teacher_answer, student_answer)

            elif q_type == "english":
                try:
                    response = requests.post(
                        "https://febinzz-miniproject.hf.space/evaluate-semantic",
                        json={
                             "question": question,
                             "teacher_answer": teacher_answer,
                             "student_answer": student_answer
                        },
                        timeout=30
                    )

                    result = response.json().get("correct", False)

                    print(
                        f"✅ English Semantic evaluated via API: "
                        f"Teacher='{teacher_answer}', "
                        f"Student='{student_answer}', "
                        f"Result={result}"
                    )
                except Exception as sem_error:
                    print(
                        f"⚠️ Semantic API failed: {sem_error}, "
                        f"falling back to string comparison"
                    )
                    result = (
                        student_answer.lower().strip()
                        == teacher_answer.lower().strip()
                    )
            else:
                return jsonify({"error": "Invalid type"}), 400

            return jsonify({"correct": result})
        
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return jsonify({"error": str(e), "correct": False}), 500

    @app.route("/")
    def index():
        # test database connectivity but always return the front page if
        # available. we log the result to the console so developers can see
        # connection problems without having to change the HTML.
        try:
            db.session.execute(text("SELECT 1"))
            app.logger.info("Database connection successful")
        except Exception as e:
            app.logger.warning(f"Database connection failed: {e}")

        # send the existing index.html in the repository root
        return app.send_static_file('index.html')

    # ------------------------------------------------------------------
    # student authentication/api endpoints
    # ------------------------------------------------------------------

    def _current_student():
        sid = session.get('student_id')
        if sid:
            return Student.query.get(sid)
        return None
    @app.route("/create-exam")
    def create_exam():
        with open("exam.html", "r", encoding="utf-8") as f:
            return f.read()
    @app.route('/api/teacher/students_for_exam', methods=['GET'])
    def students_for_exam():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401

            def _normalise_multi(values):
                items = []
                for value in values:
                    if not value:
                        continue
                    parts = [part.strip() for part in value.split(',') if part.strip()]
                    items.extend(parts)
                return [item for item in items if item.lower() != 'all']

            dept_values = _normalise_multi(request.args.getlist('department') or [request.args.get('department', 'all')])
            batch_values = _normalise_multi(request.args.getlist('batch') or [request.args.get('batch', 'all')])  # first 2 chars of email

            query = Student.query
            students = query.all()

            if dept_values:
                dept_set = {dept.upper() for dept in dept_values}
                students = [s for s in students if (s.department or '').upper() in dept_set]

            # filter by email prefix (batch) if specified
            if batch_values:
                batch_set = {batch.lower() for batch in batch_values}
                students = [s for s in students if (s.email or '').lower()[:2] in batch_set]

            result = []
            for s in students:
                email_prefix = (s.email or '')[:2].lower()
                result.append({
                    "id": s.id,
                    "name": s.name,
                    "roll_number": s.roll_number,
                    "department": s.department,
                    "semester": s.semester,
                    "email_prefix": email_prefix
                    })
            return jsonify(students=result)
        except Exception as e:
            app.logger.error(f"Get students for exam error: {e}")
            return jsonify(message='failed to fetch students'), 500
    @app.route('/api/student/register', methods=['POST'])
    def student_register():
        try:
            data = request.get_json() or {}

        # basic validation
            if not data.get('username') or not data.get('password'):
                return jsonify(message='username and password required'), 400

        # -------------------------
        # Username validation
        # only small letters + numbers
        # -------------------------
            username = data.get('username', '').strip()
            if not re.match(r'^[a-z0-9]+$', username):
                return jsonify(message='Username must contain only small letters and numbers'), 400

            email = data.get('email', '').strip()
            phone = str(data.get('phone', '')).strip()
            department = str(data.get('department', '')).strip().upper()

            if not _is_valid_email(email):
                return jsonify(message='Email must be a valid email address'), 400
            if not _is_valid_phone(phone):
                return jsonify(message='Phone number must be exactly 10 digits'), 400
            if department not in VALID_DEPARTMENTS:
                return jsonify(message='Please select a valid department'), 400

            exists = Student.query.filter(
                (Student.username == username) |
                (Student.email == email) |
                (Student.roll_number == data.get('roll_number'))
                ).first()

            if exists:
                return jsonify(message='user already exists'), 400

            student = Student(
                name=data.get('name'),
                username=username,
                email=email,
                department=department,
                phone=phone,
                roll_number=data.get('roll_number'),
                semester=data.get('semester')
                )

            student.set_password(data.get('password'))

            if data.get('photo'):
                student.photo = data.get('photo')

            db.session.add(student)
            db.session.commit()

            return jsonify(message='registered'), 200

        except Exception as e:
            app.logger.error(f"Registration error: {e}")
            return jsonify(message='registration failed'), 500
    # -------------------------
# Username validation
# -------------------------
        username = data.get('username', '').strip()
        if not re.match(r'^[a-z0-9]+$', username):
            return jsonify(message='Username must contain only small letters and numbers'), 400

# -------------------------
# Email validation
# -------------------------
        email = data.get('email', '').strip()

        if not _is_valid_email(email):
            return jsonify(message='Email must be a valid email address'), 400
    @app.route('/api/student/login', methods=['POST'])
    def student_login():
        try:
            data = request.get_json() or {}
            username = data.get('username')
            password = data.get('password')
            student = None
            if username:
                student = Student.query.filter_by(username=username).first()
            if not student and data.get('roll_number'):
                student = Student.query.filter_by(roll_number=data.get('roll_number')).first()
            if not student or not student.check_password(password):
                return jsonify(message='invalid credentials'), 401
            session['student_id'] = student.id
            return jsonify(message='logged in'), 200
        except Exception as e:
            app.logger.error(f"Login error: {e}")
            return jsonify(message='login failed'), 500

    @app.route('/api/student/logout', methods=['POST'])
    def student_logout():
        session.pop('student_id', None)
        return jsonify(message='logged out')

    @app.after_request
    def add_security_headers(response):
        # Prevent caching of authenticated pages so browser Back can't show stale protected pages
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, private'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        # Some additional security headers
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Referrer-Policy', 'no-referrer-when-downgrade')
        return response

    @app.route('/api/student/me', methods=['GET'])
    def student_me():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            return jsonify(
                id=student.id,
                username=student.username,
                name=student.name,
                email=student.email,
                department=student.department,
                phone=student.phone,
                roll_number=student.roll_number,
                semester=student.semester,
                photo=student.photo,
            )
        except Exception as e:
            app.logger.error(f"Get student info error: {e}")
            return jsonify(message='failed to get student info'), 500

    @app.route('/api/student/update', methods=['POST'])
    def student_update():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            if 'username' in data:
                username = str(data.get('username', '')).strip()
                if username != (student.username or '') and not re.match(r'^[a-z]+$', username):
                    return jsonify(message='Username must contain only lowercase letters (a-z)'), 400
                data['username'] = username
            if 'email' in data:
                email = str(data.get('email', '')).strip()
                if not _is_valid_email(email):
                    return jsonify(message='Email must be a valid email address'), 400
                data['email'] = email
            if 'phone' in data:
                phone = str(data.get('phone', '')).strip()
                if not _is_valid_phone(phone):
                    return jsonify(message='Phone number must be exactly 10 digits'), 400
                data['phone'] = phone
            if 'department' in data:
                department = str(data.get('department', '')).strip().upper()
                if department not in VALID_DEPARTMENTS:
                    return jsonify(message='Please select a valid department'), 400
                data['department'] = department
            if 'roll_number' in data:
                roll_number = str(data.get('roll_number', '')).strip()
                if not roll_number:
                    return jsonify(message='roll number required'), 400
                if len(roll_number) > 50:
                    return jsonify(message='roll number is too long (max 50 characters)'), 400
                exists = Student.query.filter(
                    Student.roll_number == roll_number,
                    Student.id != student.id
                ).first()
                if exists:
                    return jsonify(message='roll number already exists'), 400
                data['roll_number'] = roll_number
            # only update known fields
            for field in ('name', 'email', 'department', 'phone', 'roll_number', 'semester', 'photo', 'username'):
                if field in data:
                    setattr(student, field, data[field])
            db.session.commit()
            return jsonify(message='updated')
        except IntegrityError:
            db.session.rollback()
            return jsonify(message='username/email/roll number already exists'), 400
        except Exception as e:
            db.session.rollback()
            app.logger.exception("Update student error")
            return jsonify(message=f'update failed: {str(e)}'), 500

    @app.route('/api/student/change_password', methods=['POST'])
    def student_change_password():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            current = data.get('current_password')
            new = data.get('new_password')
            if not student.check_password(current):
                return jsonify(message='incorrect current password'), 400
            student.set_password(new)
            db.session.commit()
            return jsonify(message='password changed')
        except Exception as e:
            app.logger.error(f"Change password error: {e}")
            return jsonify(message='password change failed'), 500

    # -------------------- OTP Forgot / Reset Password --------------------
    def _otp_expiry_minutes():
        return 10

    def _generate_otp():
        return f"{secrets.randbelow(1000000):06d}"

    def _send_smtp_email(email: str, subject: str, body: str):
        """Send a plain-text email through Gmail SMTP."""
        username = app.config.get('MAIL_USERNAME')
        password = app.config.get('MAIL_PASSWORD')
        sender = app.config.get('MAIL_DEFAULT_SENDER') or username

        app.logger.info(f"SMTP email sending started for: {email}")

        if not username or not password:
            app.logger.error('MAIL_USERNAME or MAIL_PASSWORD is not configured')
            return False

        if not sender or '@' not in sender:
            app.logger.error('MAIL_DEFAULT_SENDER must be a valid email address')
            return False

        message = MIMEText(body, 'plain')
        message['Subject'] = subject
        message['From'] = sender
        message['To'] = email

        try:
            app.logger.info('Connecting to Gmail SMTP...')
            context = ssl.create_default_context()
            with smtplib.SMTP(app.config.get('MAIL_SERVER', 'smtp.gmail.com'), app.config.get('MAIL_PORT', 587)) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(username, password)
                server.sendmail(sender, [email], message.as_string())
            app.logger.info(f'SMTP email sent successfully to {email}')
            return True
        except smtplib.SMTPAuthenticationError as exc:
            app.logger.error(f'Gmail SMTP authentication failed: {exc}')
            return False
        except smtplib.SMTPException as exc:
            app.logger.error(f'Gmail SMTP error while sending email to {email}: {exc}')
            return False
        except Exception as exc:
            app.logger.error(f'Unexpected email sending error for {email}: {exc}')
            return False

    def send_otp_email(email: str, otp: str):
        """Send a plain-text OTP email through Gmail SMTP.

        Gmail App Password setup:
        1. Enable 2-Step Verification on your Google account.
        2. Open https://myaccount.google.com/apppasswords
        3. Create an App Password for Mail.
        4. Put the generated 16-character password in MAIL_PASSWORD.
        """
        app.logger.info(f"OTP email sending started for: {email}")

        body = (
            f'Your OTP for password reset is: {otp}\n\n'
            'This OTP will expire in 10 minutes.\n\n'
            'If you did not request this, ignore this email.'
        )

        return _send_smtp_email(email, 'Password Reset OTP', body)

    def send_username_email(email: str, username_text: str):
        """Send a username recovery email through Gmail SMTP."""
        app.logger.info(f"Username email sending started for: {email}")
        body = (
            'Your username for the Student Management System is:\n\n'
            f'{username_text}\n\n'
            'If you did not request this email, please ignore it.'
        )
        return _send_smtp_email(email, 'Username Recovery', body)

    def _find_user_by_email(email: str):
        student = Student.query.filter_by(email=email).first()
        if student:
            return student
        teacher = Teacher.query.filter_by(email=email).first()
        if teacher:
            return teacher
        return None

    @app.route('/api/send-otp', methods=['POST'])
    def send_otp():
        """Generate a 6-digit OTP and send it to the entered email."""
        try:
            data = request.get_json() or {}
            email = (data.get('email') or '').strip().lower()
            if not _is_valid_email(email):
                return jsonify(success=False, message='Invalid email'), 400
            # Only allow sending OTP if the email belongs to a registered user
            user = _find_user_by_email(email)
            if not user:
                app.logger.info(f"OTP request refused: no account for {email}")
                return jsonify(success=False, message='Email not registered'), 400

            otp = _generate_otp()
            expiry = datetime.utcnow() + timedelta(minutes=_otp_expiry_minutes())
            app.logger.info(f"OTP generated for: {email}")

            user.otp_code = otp
            user.otp_expiry = expiry
            db.session.commit()
            app.logger.info('OTP stored successfully')

            sent = send_otp_email(email, otp)
            if not sent:
                user.otp_code = None
                user.otp_expiry = None
                db.session.commit()
                return jsonify(success=False, message='OTP could not be sent'), 500
            return jsonify(success=True, message='OTP sent successfully'), 200
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Email sending failure: {e}")
            return jsonify(success=False, message='OTP could not be sent'), 500

    @app.route('/api/verify-otp', methods=['POST'])
    def verify_otp():
        """Verify OTP and update the password for either a student or teacher."""
        try:
            data = request.get_json() or {}
            email = (data.get('email') or '').strip().lower()
            otp = (data.get('otp') or '').strip()
            password = (data.get('password') or '').strip()

            if not _is_valid_email(email):
                return jsonify(success=False, message='Invalid email'), 400
            if not re.match(r'^\d{6}$', otp):
                return jsonify(success=False, message='Invalid or expired OTP'), 400
            if len(password) < 6:
                return jsonify(success=False, message='Password must be at least 6 characters'), 400

            user = _find_user_by_email(email)
            if not user or not user.otp_code or not user.otp_expiry:
                return jsonify(success=False, message='Invalid or expired OTP'), 400

            if user.otp_code != otp or user.otp_expiry < datetime.utcnow():
                return jsonify(success=False, message='Invalid or expired OTP'), 400

            user.set_password(password)
            user.otp_code = None
            user.otp_expiry = None
            db.session.commit()
            return jsonify(success=True, message='Password updated successfully'), 200
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"OTP verification error: {e}")
            return jsonify(success=False, message='Invalid or expired OTP'), 400

    @app.route('/api/forgot-username', methods=['POST'])
    def forgot_username():
        """Send the username to the entered email if a matching account exists."""
        try:
            data = request.get_json() or {}
            email = (data.get('email') or '').strip().lower()
            if not _is_valid_email(email):
                return jsonify(success=False, message='Invalid email'), 400
            app.logger.info(f'Username recovery requested for: {email}')

            user = Student.query.filter_by(email=email).first()
            if not user:
                user = Teacher.query.filter_by(email=email).first()

            if not user or not user.username:
                app.logger.info(f'Username recovery refused: no account for {email}')
                return jsonify(success=False, message='Email not registered'), 400

            sent = send_username_email(email, user.username)
            if sent:
                app.logger.info('Username email sent successfully')
                return jsonify(success=True, message='Username sent successfully'), 200
            else:
                app.logger.error('Username email failed')
                return jsonify(success=False, message='Failed to send username'), 500
        except Exception as e:
            app.logger.error(f'Username recovery error: {e}')
            return jsonify(success=True, message='If the email exists, the username has been sent.'), 200

    # ------------------------------------------------------------------
    # teacher authentication/api endpoints
    # ------------------------------------------------------------------

    def _current_teacher():
        tid = session.get('teacher_id')
        if tid:
            return Teacher.query.get(tid)
        return None

    @app.route('/api/teacher/register', methods=['POST'])
    def teacher_register():
        try:
            data = request.get_json() or {}
            if not data.get('username') or not data.get('password'):
                return jsonify(message='username and password required'), 400

            email = str(data.get('email', '')).strip()
            phone = str(data.get('phone', '')).strip()
            department = str(data.get('department', '')).strip().upper()

            if not _is_valid_email(email):
                return jsonify(message='Email must be a valid email address'), 400
            if not _is_valid_phone(phone):
                return jsonify(message='Phone number must be exactly 10 digits'), 400
            if department not in VALID_DEPARTMENTS:
                return jsonify(message='Please select a valid department'), 400

            exists = Teacher.query.filter(
                (Teacher.username == data.get('username')) |
                (Teacher.email == email)
            ).first()
            if exists:
                return jsonify(message='user already exists'), 400

            teacher = Teacher(
                name=data.get('name'),
                username=data.get('username'),
                email=email,
                department=department,
                phone=phone
            )
            teacher.set_password(data.get('password'))
            db.session.add(teacher)
            db.session.commit()
            return jsonify(message='registered'), 200
        except Exception as e:
            app.logger.error(f"Teacher registration error: {e}")
            return jsonify(message='registration failed'), 500

    @app.route('/api/teacher/login', methods=['POST'])
    def teacher_login():
        try:
            data = request.get_json() or {}
            username = data.get('username')
            password = data.get('password')
            teacher = None
            if username:
                teacher = Teacher.query.filter_by(username=username).first()
            if not teacher or not teacher.check_password(password):
                return jsonify(message='invalid credentials'), 401
            session['teacher_id'] = teacher.id
            return jsonify(message='logged in'), 200
        except Exception as e:
            app.logger.error(f"Teacher login error: {e}")
            return jsonify(message='login failed'), 500

    @app.route('/api/teacher/logout', methods=['POST'])
    def teacher_logout():
        session.pop('teacher_id', None)
        return jsonify(message='logged out')

    @app.route('/api/teacher/me', methods=['GET'])
    def teacher_me():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            return jsonify(
                id=teacher.id,
                username=teacher.username,
                name=teacher.name,
                email=teacher.email,
                department=teacher.department,
                phone=teacher.phone,
                photo=teacher.photo,
            )
        except Exception as e:
            app.logger.error(f"Get teacher info error: {e}")
            return jsonify(message='failed to get teacher info'), 500

    @app.route('/api/teacher/update', methods=['POST'])
    def teacher_update():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            if 'username' in data:
                username = str(data.get('username', '')).strip()
                if username != (teacher.username or '') and not re.match(r'^[a-z]+$', username):
                    return jsonify(message='Username must contain only lowercase letters (a-z)'), 400
                data['username'] = username
            if 'email' in data:
                email = str(data.get('email', '')).strip()
                if not _is_valid_email(email):
                    return jsonify(message='Email must be a valid email address'), 400
                data['email'] = email
            if 'phone' in data:
                phone = str(data.get('phone', '')).strip()
                if not _is_valid_phone(phone):
                    return jsonify(message='Phone number must be exactly 10 digits'), 400
                data['phone'] = phone
            if 'department' in data:
                department = str(data.get('department', '')).strip().upper()
                if department not in VALID_DEPARTMENTS:
                    return jsonify(message='Please select a valid department'), 400
                data['department'] = department

            for field in ('name', 'email', 'department', 'phone', 'username', 'photo'):
                if field in data:
                    setattr(teacher, field, data[field])
            db.session.commit()
            return jsonify(message='updated')
        except IntegrityError:
            db.session.rollback()
            return jsonify(message='username/email already exists'), 400
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Update teacher error: {e}")
            return jsonify(message='update failed'), 500

    @app.route('/api/teacher/change_password', methods=['POST'])
    def teacher_change_password():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            current = data.get('current_password')
            new = data.get('new_password')
            if not teacher.check_password(current):
                return jsonify(message='incorrect current password'), 400
            teacher.set_password(new)
            db.session.commit()
            return jsonify(message='password changed')
        except Exception as e:
            app.logger.error(f"Change teacher password error: {e}")
            return jsonify(message='password change failed'), 500
        # -------------------------------------------------------------
    # GET ALL STUDENTS (for Teacher Dashboard)
    # -------------------------------------------------------------
    @app.route('/api/teacher/students', methods=['GET'])
    def teacher_get_students():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401

            students = Student.query.all()

            result = []
            for s in students:
                result.append({
                    "id": s.id,
                    "name": s.name,
                    "username": s.username,
                    "email": s.email,
                    "department": s.department,
                    "phone": s.phone,
                    "roll_number": s.roll_number,
                    "semester": s.semester
                })

            return jsonify(students=result)

        except Exception as e:
            app.logger.error(f"Get students error: {e}")
            return jsonify(message='failed to fetch students'), 500

    def _to_dt(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _exam_to_dict(exam, include_questions=False, include_students=False):
        out = {
            'id': exam.id,
            'name': exam.name,
            'startTime': exam.start_time.isoformat() if exam.start_time else None,
            'endTime': exam.end_time.isoformat() if exam.end_time else None,
            'duration': exam.duration,
            'active': bool(exam.status_active),
            'questions': []
        }
        if include_questions:
            qs = Question.query.filter_by(exam_id=exam.id).order_by(Question.id.asc()).all()
            out['questions'] = [
                {
                    'id': q.id,
                    'question': q.question_text,
                    'teacherAnswer': q.correct_answer,
                    'type': (getattr(q, 'question_type', None) or 'direct')
                }
                for q in qs
            ]
        if include_students:
            links = ExamStudent.query.filter_by(exam_id=exam.id).all()
            rows = []
            for link in links:
                student = Student.query.get(link.student_id)
                if student:
                    rows.append({
                        'id': student.id,
                        'roll_number': student.roll_number,
                        'name': student.name
                    })
            out['students'] = rows
        return out

    def _refresh_exam_status(exam):
        now = app_now()
        should_be_active = bool(exam.start_time and exam.end_time and exam.start_time <= now <= exam.end_time)
        if exam.status_active != should_be_active:
            exam.status_active = should_be_active

    def _mark_missed_exam_students(exam, student_id=None):
        if not exam or not exam.end_time:
            return False

        now = app_now()
        if now < exam.end_time:
            return False

        query = ExamStudent.query.filter_by(exam_id=exam.id)
        if student_id is not None:
            query = query.filter_by(student_id=student_id)

        links = query.filter(ExamStudent.status.in_(['not-attempted', 'in-progress'])).all()
        changed = False
        for link in links:
            link.status = 'missed'
            if not link.end_time:
                link.end_time = exam.end_time
            changed = True
        return changed

    @app.route('/api/teacher/exams', methods=['GET'])
    def teacher_get_exams():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401

            exams = Exam.query.filter_by(created_by=teacher.id).order_by(Exam.id.desc()).all()
            for exam in exams:
                _refresh_exam_status(exam)
                _mark_missed_exam_students(exam)
            db.session.commit()

            return jsonify(exams=[_exam_to_dict(exam, include_students=True, include_questions=True) for exam in exams])
        except Exception as e:
            app.logger.error(f"Teacher get exams error: {e}")
            return jsonify(message='failed to fetch exams'), 500

    @app.route('/api/teacher/exams', methods=['POST'])
    def teacher_create_exam():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401

            data = request.get_json() or {}
            student_roll_numbers = data.get('student_roll_numbers') or []
            exam_name = (data.get('name') or '').strip()
            if not exam_name:
                return jsonify(message='exam name required'), 400

            exam = Exam(
                name=exam_name,
                start_time=None,
                end_time=None,
                duration=0,
                created_by=teacher.id,
                status_active=False,
                questions_count=0,
            )

            db.session.add(exam)
            db.session.flush()

            students = Student.query.filter(Student.roll_number.in_(student_roll_numbers)).all() if student_roll_numbers else []
            for s in students:
                db.session.add(ExamStudent(exam_id=exam.id, student_id=s.id, status='not-attempted', score=0, answers=[]))

            db.session.commit()
            return jsonify(message='created', exam=_exam_to_dict(exam, include_students=True)), 201
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher create exam error: {e}")
            return jsonify(message='failed to create exam'), 500

    @app.route('/api/teacher/exams/<int:exam_id>/questions', methods=['POST'])
    def teacher_save_exam_questions(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401

            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404

            now = app_now()
            if exam.end_time and now >= exam.end_time:
                return jsonify(message='exam already ended; questions cannot be edited now'), 400

            data = request.get_json() or {}
            questions = data.get('questions') or []
            schedule_start = _to_dt(data.get('startTime'))
            schedule_end = _to_dt(data.get('endTime'))
            schedule_duration = int(data.get('duration') or 0)

            if not isinstance(questions, list) or len(questions) == 0:
                return jsonify(message='questions required'), 400
            if not schedule_start or not schedule_end or schedule_end <= schedule_start or schedule_duration <= 0:
                return jsonify(message='valid start/end time and individual duration are required'), 400

            Question.query.filter_by(exam_id=exam.id).delete()
            for item in questions:
                db.session.add(Question(
                    exam_id=exam.id,
                    question_text=(item.get('question') or '').strip(),
                    correct_answer=(item.get('teacherAnswer') or '').strip(),
                    question_type=(item.get('type') or 'direct').strip()
                ))

            exam.questions_count = len(questions)
            exam.start_time = schedule_start
            exam.end_time = schedule_end
            exam.duration = schedule_duration
            exam.status_active = False
            db.session.commit()
            return jsonify(message='saved')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher save questions error: {e}")
            return jsonify(message='failed to save questions'), 500

    @app.route('/api/teacher/exams/<int:exam_id>/activate', methods=['POST'])
    def teacher_activate_exam(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404
            exam.status_active = True
            db.session.commit()
            return jsonify(message='activated')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher activate exam error: {e}")
            return jsonify(message='failed to activate exam'), 500

    @app.route('/api/teacher/exams/<int:exam_id>/deactivate', methods=['POST'])
    def teacher_deactivate_exam(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404
            exam.status_active = False
            db.session.commit()
            return jsonify(message='deactivated')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher deactivate exam error: {e}")
            return jsonify(message='failed to deactivate exam'), 500

    @app.route('/api/teacher/exams/<int:exam_id>/extend', methods=['POST'])
    def teacher_extend_exam(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404

            now = app_now()
            if not exam.start_time or not exam.end_time:
                return jsonify(message='exam schedule not set'), 400
            if now < exam.start_time:
                return jsonify(message='exam not started yet'), 400
            if now >= exam.end_time:
                return jsonify(message='exam already ended; cannot extend'), 400

            data = request.get_json() or {}
            new_end = _to_dt(data.get('endTime'))
            if not new_end or (exam.end_time and new_end <= exam.end_time):
                return jsonify(message='invalid end time'), 400
            exam.end_time = new_end
            if exam.start_time:
                exam.duration = max(1, int((exam.end_time - exam.start_time).total_seconds() / 60))
            db.session.commit()
            return jsonify(message='extended')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher extend exam error: {e}")
            return jsonify(message='failed to extend exam'), 500

    @app.route('/api/teacher/exams/<int:exam_id>/schedule', methods=['POST'])
    def teacher_schedule_exam(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404
            data = request.get_json() or {}
            new_start = _to_dt(data.get('startTime'))
            new_end = _to_dt(data.get('endTime'))
            if not new_start or not new_end or new_end <= new_start:
                return jsonify(message='invalid schedule'), 400
            exam.start_time = new_start
            exam.end_time = new_end
            exam.duration = max(1, int((new_end - new_start).total_seconds() / 60))
            exam.status_active = False
            db.session.commit()
            return jsonify(message='scheduled')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher schedule exam error: {e}")
            return jsonify(message='failed to schedule exam'), 500

    @app.route('/api/teacher/exams/<int:exam_id>', methods=['DELETE'])
    def teacher_delete_exam(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404
            Question.query.filter_by(exam_id=exam.id).delete()
            ExamStudent.query.filter_by(exam_id=exam.id).delete()
            db.session.delete(exam)
            db.session.commit()
            return jsonify(message='deleted')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher delete exam error: {e}")
            return jsonify(message='failed to delete exam'), 500

    @app.route('/api/teacher/exams/<int:exam_id>/report', methods=['GET'])
    def teacher_exam_report(exam_id):
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam = Exam.query.filter_by(id=exam_id, created_by=teacher.id).first()
            if not exam:
                return jsonify(message='exam not found'), 404

            _refresh_exam_status(exam)
            _mark_missed_exam_students(exam)

            questions = Question.query.filter_by(exam_id=exam.id).order_by(Question.id.asc()).all()
            q_payload = [
                {
                    'question': q.question_text,
                    'teacherAnswer': q.correct_answer,
                    'type': (getattr(q, 'question_type', None) or 'direct')
                }
                for q in questions
            ]

            rows = []
            links = ExamStudent.query.filter_by(exam_id=exam.id).all()
            for link in links:
                student = Student.query.get(link.student_id)
                if not student:
                    continue
                answers = link.answers or {}
                student_answers = answers.get('studentAnswers') or []
                question_results = answers.get('questionResults') or []
                submitted = link.status == 'completed'
                rows.append({
                    'student_roll': student.roll_number,
                    'student_name': student.name,
                    'submitted': submitted,
                    'score': link.score or 0,
                    'total_questions': len(q_payload),
                    'questions': q_payload,
                    'studentAnswers': student_answers,
                    'questionResults': question_results,
                    'missed': link.status == 'missed'
                })
            db.session.commit()
            return jsonify(exam={'id': exam.id, 'name': exam.name}, rows=rows)
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher exam report error: {e}")
            return jsonify(message='failed to fetch report'), 500

    @app.route('/api/teacher/results/clear', methods=['POST'])
    def teacher_clear_results():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            exam_ids = [e.id for e in Exam.query.filter_by(created_by=teacher.id).all()]
            if exam_ids:
                links = ExamStudent.query.filter(ExamStudent.exam_id.in_(exam_ids)).all()
                for link in links:
                    link.status = 'not-attempted'
                    link.score = 0
                    link.answers = []
                    link.start_time = None
                    link.end_time = None
                db.session.commit()
            return jsonify(message='cleared')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Teacher clear results error: {e}")
            return jsonify(message='failed to clear results'), 500

    @app.route('/api/student/exams', methods=['GET'])
    def student_get_exams():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401

            links = ExamStudent.query.filter_by(student_id=student.id).all()
            out = []
            for link in links:
                exam = Exam.query.get(link.exam_id)
                if not exam:
                    continue
                _refresh_exam_status(exam)
                _mark_missed_exam_students(exam, student_id=student.id)
                out.append({
                    'id': exam.id,
                    'name': exam.name,
                    'startTime': exam.start_time.isoformat() if exam.start_time else None,
                    'endTime': exam.end_time.isoformat() if exam.end_time else None,
                    'duration': exam.duration,
                    'active': bool(exam.status_active),
                    'submitted': link.status == 'completed',
                    'missed': link.status == 'missed',
                    'score': link.score,
                    'questions': []
                })
            db.session.commit()
            return jsonify(exams=out)
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Student get exams error: {e}")
            return jsonify(message='failed to fetch exams'), 500

    @app.route('/api/student/exams/<int:exam_id>/start', methods=['POST'])
    def student_start_exam(exam_id):
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            link = ExamStudent.query.filter_by(exam_id=exam_id, student_id=student.id).first()
            exam = Exam.query.get(exam_id)
            if not exam or not link:
                return jsonify(message='exam not found'), 404

            _refresh_exam_status(exam)
            if not exam.status_active:
                db.session.commit()
                return jsonify(message='exam is not active'), 400

            if link.status == 'completed':
                db.session.commit()
                return jsonify(message='already submitted'), 400

            if not link.start_time:
                link.start_time = app_now()
                link.status = 'in-progress'
            questions = Question.query.filter_by(exam_id=exam.id).order_by(Question.id.asc()).all()
            db.session.commit()
            return jsonify(exam={
                'id': exam.id,
                'name': exam.name,
                'startTime': exam.start_time.isoformat() if exam.start_time else None,
                'endTime': exam.end_time.isoformat() if exam.end_time else None,
                'duration': exam.duration,
                'questions': [
                    {
                        'id': q.id,
                        'question': q.question_text,
                        'teacherAnswer': q.correct_answer,
                        'type': (getattr(q, 'question_type', None) or 'direct')
                    }
                    for q in questions
                ]
            })
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Student start exam error: {e}")
            return jsonify(message='failed to start exam'), 500

    @app.route('/api/student/exams/<int:exam_id>/submit', methods=['POST'])
    def student_submit_exam(exam_id):
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            link = ExamStudent.query.filter_by(exam_id=exam_id, student_id=student.id).first()
            if not link:
                return jsonify(message='exam not found'), 404

            data = request.get_json() or {}
            link.status = 'completed'
            link.score = int(data.get('score') or 0)
            link.answers = {
                'studentAnswers': data.get('studentAnswers') or [],
                'questionResults': data.get('questionResults') or []
            }
            link.end_time = app_now()
            db.session.commit()
            return jsonify(message='submitted')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Student submit exam error: {e}")
            return jsonify(message='failed to submit exam'), 500

    @app.route('/api/student/results', methods=['GET'])
    def student_results():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            links = ExamStudent.query.filter_by(student_id=student.id).all()
            now = app_now()
            rows = []
            for link in links:
                exam = Exam.query.get(link.exam_id)
                if exam:
                    _refresh_exam_status(exam)
                    _mark_missed_exam_students(exam, student_id=student.id)
                if not exam:
                    continue

                is_missed = (
                    link.status == 'missed' or
                    (
                        link.status in ('not-attempted', 'in-progress') and
                        exam.end_time is not None and
                        now >= exam.end_time
                    )
                )
                if is_missed and link.status != 'missed':
                    link.status = 'missed'
                    if not link.end_time:
                        link.end_time = exam.end_time

                if link.status != 'completed' and not is_missed:
                    continue

                questions = Question.query.filter_by(exam_id=exam.id).order_by(Question.id.asc()).all()
                answers = link.answers or {}
                rows.append({
                    'id': exam.id,
                    'name': exam.name,
                    'startTime': exam.start_time.isoformat() if exam.start_time else None,
                    'endTime': exam.end_time.isoformat() if exam.end_time else None,
                    'duration': exam.duration,
                    'submitted': link.status == 'completed',
                    'missed': is_missed,
                    'score': link.score or 0,
                    'questions': [
                        {
                            'question': q.question_text,
                            'teacherAnswer': q.correct_answer,
                            'type': (getattr(q, 'question_type', None) or 'direct')
                        }
                        for q in questions
                    ],
                    'studentAnswers': answers.get('studentAnswers') or [],
                    'questionResults': answers.get('questionResults') or []
                })
            db.session.commit()
            return jsonify(results=rows)
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Student results error: {e}")
            return jsonify(message='failed to fetch results'), 500

    @app.route('/api/student/results/clear', methods=['POST'])
    def student_clear_results():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            links = ExamStudent.query.filter_by(student_id=student.id).all()
            for link in links:
                link.status = 'not-attempted'
                link.score = 0
                link.answers = []
                link.start_time = None
                link.end_time = None
            db.session.commit()
            return jsonify(message='cleared')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Student clear results error: {e}")
            return jsonify(message='failed to clear results'), 500

    return app


def create_tables(app=None):
    """Create all database tables.

    If an app is not provided, a new one will be created using :func:`create_app`.
    This function should be called from a context where the application
    configuration has been loaded (for example, from a script or a shell).

    The database URI may point to a MySQL server that is unreachable or
    requires credentials we don't have; in that case we automatically fall
    back to a local SQLite file so the rest of the application can still run.
    """
    if app is None:
        app = create_app()

    # flask-sqlalchemy needs an application context for create_all
    with app.app_context():
        try:
            # first attempt to create tables using whatever URI is configured
            db.create_all()
        except Exception as exc:
            # if there was any problem (e.g. access denied) log it and
            # switch to sqlite fallback before trying once more.
            app.logger.warning(f"primary create_all failed: {exc}")

            fallback = "sqlite:///fallback.db"
            app.logger.info(f"switching to fallback database {fallback}")
            app.config["SQLALCHEMY_DATABASE_URI"] = fallback
            # dispose existing engine so a new one is created using the changed
            # configuration; ``db.init_app`` should not be called again.
            try:
                db.engine.dispose()
            except Exception:
                pass
            try:
                db.create_all()
            except Exception as exc2:
                # give up after second failure
                app.logger.error(f"fallback create_all also failed: {exc2}")
        # ensure additional columns introduced after initial schema creation
        # (e.g. roll_number, semester, photo on Student) exist.
        try:
            from sqlalchemy import inspect, text
            insp = inspect(db.engine)
            if 'student' in insp.get_table_names():
                cols = {c['name'] for c in insp.get_columns('student')}
                with db.engine.begin() as conn:
                    if 'roll_number' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN roll_number VARCHAR(50)'))
                    if 'semester' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN semester VARCHAR(20)'))
                    if 'photo' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN photo TEXT'))
                    if 'otp_code' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN otp_code VARCHAR(6)'))
                    if 'otp_expiry' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN otp_expiry DATETIME'))
            if 'teacher' in insp.get_table_names():
                tcols = {c['name'] for c in insp.get_columns('teacher')}
                with db.engine.begin() as conn:
                    if 'photo' not in tcols:
                        conn.execute(text('ALTER TABLE teacher ADD COLUMN photo TEXT'))
                    if 'otp_code' not in tcols:
                        conn.execute(text('ALTER TABLE teacher ADD COLUMN otp_code VARCHAR(6)'))
                    if 'otp_expiry' not in tcols:
                        conn.execute(text('ALTER TABLE teacher ADD COLUMN otp_expiry DATETIME'))
            if 'question' in insp.get_table_names():
                qcols = {c['name'] for c in insp.get_columns('question')}
                with db.engine.begin() as conn:
                    if 'question_type' not in qcols:
                        conn.execute(text("ALTER TABLE question ADD COLUMN question_type VARCHAR(30) DEFAULT 'direct'"))
        except Exception as exc:
            app.logger.warning(f"could not ensure student columns: {exc}")


# allow command-line invocation for convenience
# running this module directly will create any missing tables and then
# start the development server so you can browse the app at
# http://localhost:5000 (or 0.0.0.0 inside the container).
def _kill_other_instances():
    """Terminate any previously running copies of this script.

    This is useful in a development container where `python app.py` may be
    invoked repeatedly and older processes linger, causing "address already
    in use" errors.  We use ``psutil`` to identify other ``app.py`` commands
    and kill them.  If the library is unavailable we simply return silently.
    """
    try:
        import psutil
        import os
        me = os.getpid()
        for proc in psutil.process_iter(['pid', 'cmdline']):
            info = proc.info
            if info['pid'] == me:
                continue
            cmd = info.get('cmdline') or []
            # match on filename to avoid killing unrelated processes
            if any('app.py' in part for part in cmd):
                proc.kill()
    except ImportError:
        # psutil not installed; nothing we can do
        pass
 

if __name__ == "__main__":
    # clean up any earlier runs so the port will be free
    _kill_other_instances()

    application = create_app()
    create_tables(application)
    print("Tables created (if they didn't exist already). Starting server...")
    # `debug=True` is fine for development; turn off in production.
    port = int(os.getenv("PORT", 5000))
    # disable the built-in reloader; it spawns a second process which
    # our _kill_other_instances helper may immediately kill, leading to the
    # mysterious exit-137 behaviour observed in the container.  The dev
    # server will still restart when the file changes if `debug=True` but
    # without the extra process.
    application.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
