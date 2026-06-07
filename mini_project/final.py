import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ----------------------------
# LOAD NLP MODELS (ONCE)
# ----------------------------
sbert = SentenceTransformer("all-MiniLM-L6-v2")
nli = pipeline("text-classification", model="roberta-large-mnli")

# ----------------------------s
# HELPER FUNCTIONS (CODE EVAL)
# ----------------------------

def normalize(code: str) -> str:
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    return re.sub(r'\s+', '', code)


def evaluate_code_question(question, correct_blank, student_answer):
    student_n = normalize(student_answer)
    answer_n = normalize(correct_blank)

    if student_n == answer_n:
        return True

    match = re.search(r"_+", question)
    if not match:
        return False

    before = normalize(question[:match.start()])
    after = normalize(question[match.end():])
    expected_full = before + answer_n + after

    return student_n == expected_full


# ----------------------------
# DIRECT FILL-IN-THE-BLANK
# ----------------------------

def strip_context(student, context_words, from_start=True):
    tokens = student.split()

    for i in range(1, len(context_words) + 1):
        window = context_words[-i:] if from_start else context_words[:i]
        window_str = " ".join(window)

        if from_start and student.startswith(window_str):
            tokens = tokens[len(window):]
        elif not from_start and student.endswith(window_str):
            tokens = tokens[:-len(window)]

        student = " ".join(tokens).strip()

    return student


def evaluate_fill_blank(question, teacher_answer, student_answer):
    question = question.lower().strip()
    teacher_answer = teacher_answer.lower().strip()
    student_answer = student_answer.lower().strip()

    if student_answer == teacher_answer:
        return True

    parts = re.split(r"_+", question)
    left = parts[0].strip() if len(parts) > 0 else ""
    right = parts[1].strip() if len(parts) > 1 else ""

    cleaned = student_answer

    if left:
        cleaned = strip_context(cleaned, left.split(), True)
    if right:
        cleaned = strip_context(cleaned, right.split(), False)

    return cleaned == teacher_answer


# ----------------------------
# ENGLISH / SEMANTIC EVALUATION
# ----------------------------

def evaluate_semantic(question, teacher_answer, student_answer, sim_threshold=0.65):

    question = question.lower().strip()
    teacher_answer = teacher_answer.lower().strip()
    student_answer = student_answer.lower().strip()

    teacher_full = re.sub(r"_+", teacher_answer, question, count=1)
    student_full = re.sub(r"_+", student_answer, question, count=1)

    # ---- NLI CHECK ----
    nli_input = f"{teacher_full} </s></s> {student_full}"
    nli_result = nli(nli_input)[0]

    if nli_result["label"] == "CONTRADICTION" and nli_result["score"] > 0.6:
        return False

    if nli_result["label"] == "ENTAILMENT" and nli_result["score"] > 0.6:
        return True

    # ---- SBERT FALLBACK ----
    emb_teacher = sbert.encode(teacher_full, convert_to_tensor=True)
    emb_student = sbert.encode(student_full, convert_to_tensor=True)

    similarity = util.cos_sim(emb_teacher, emb_student).item()

    return similarity >= sim_threshold


# ----------------------------
# MAIN PROGRAM
# ----------------------------

def main():
    score = 0

    total_questions = int(input("Enter total number of questions: "))
    num_q = int(input("Number of NUMERICAL questions: "))
    code_q = int(input("Number of CODE questions: "))
    direct_q = int(input("Number of DIRECT fill questions: "))
    english_q = int(input("Number of ENGLISH (semantic) questions: "))

    if num_q + code_q + direct_q + english_q != total_questions:
        raise ValueError("Question count mismatch")

    # ---- NUMERICAL ----
    for i in range(num_q):
        print(f"\nNUMERICAL QUESTION {i+1}")
        input("Question: ")
        t = float(input("Teacher answer: "))
        s = float(input("Student answer: "))
        if round(t, 1) == round(s, 1):
            print("✔ Correct")
            score += 1
        else:
            print("✖ Wrong")

    # ---- CODE ----
    for i in range(code_q):
        print(f"\nCODE QUESTION {i+1}")
        q = input("Question (___ / .....): ")
        t = input("Teacher code: ")
        s = input("Student code: ")
        if evaluate_code_question(q, t, s):
            print("✔ Correct")
            score += 1
        else:
            print("✖ Wrong")

    # ---- DIRECT FILL ----
    for i in range(direct_q):
        print(f"\nDIRECT FILL QUESTION {i+1}")
        q = input("Question (___): ")
        t = input("Teacher answer: ")
        s = input("Student answer: ")
        if evaluate_fill_blank(q, t, s):
            print("✔ Correct")
            score += 1
        else:
            print("✖ Wrong")

    # ---- ENGLISH / SEMANTIC ----
    for i in range(english_q):
        print(f"\nENGLISH QUESTION {i+1}")
        q = input("Question (____): ")
        t = input("Teacher answer: ")
        s = input("Student answer: ")

        if evaluate_semantic(q, t, s):
            print("✔ Correct (Semantic Match)")
            score += 1
        else:
            print("✖ Wrong (Semantic Mismatch)")

    # ---- FINAL SCORE ----
    print("\n==============================")
    print(f"FINAL SCORE: {score} / {total_questions}")
    print("==============================")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()