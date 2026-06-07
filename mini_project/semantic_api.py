import os
import re
import traceback
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from flask_cors import CORS
# Allow skipping heavy model load in development/Spaces if needed
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
                                                                    traceback.print_exc()
                                                                            sbert = None
                                                                                    nli = None


                                                                                    def evaluate_semantic(question, teacher_answer, student_answer, sim_threshold=0.65):
                                                                                        """Exact semantic evaluation logic copied from the main app.

                                                                                            Returns True if the student answer should be considered correct.
                                                                                                Falls back to lowercase string comparison when models are unavailable.
                                                                                                    """
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


                                                                                                                                                                            def create_app():
                                                                                                                                                                                app = Flask(__name__)
                                                                                                                                                                                    CORS(app)
                                                                                                                                                                                        @app.route("/")
                                                                                                                                                                                            def home():
                                                                                                                                                                                                    return jsonify({"message": "Semantic API running"})
                                                                                                                                                                                                        @app.route('/evaluate-semantic', methods=['POST'])
                                                                                                                                                                                                            def evaluate_endpoint():
                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                data = request.get_json() or {}
                                                                                                                                                                                                                                            question = data.get('question', '')
                                                                                                                                                                                                                                                        teacher_answer = data.get('teacher_answer', '')
                                                                                                                                                                                                                                                                    student_answer = data.get('student_answer', '')

                                                                                                                                                                                                                                                                                # Basic validation
                                                                                                                                                                                                                                                                                            if question is None or teacher_answer is None or student_answer is None:
                                                                                                                                                                                                                                                                                                            return jsonify(error='Missing fields'), 400

                                                                                                                                                                                                                                                                                                                        correct = False
                                                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                                                                    correct = evaluate_semantic(question, teacher_answer, student_answer)
                                                                                                                                                                                                                                                                                                                                                                except Exception as eval_err:
                                                                                                                                                                                                                                                                                                                                                                                # If evaluation fails for any reason, fall back to simple compare
                                                                                                                                                                                                                                                                                                                                                                                                print(f"Semantic evaluation error: {eval_err}")
                                                                                                                                                                                                                                                                                                                                                                                                                traceback.print_exc()
                                                                                                                                                                                                                                                                                                                                                                                                                                correct = student_answer.lower().strip() == teacher_answer.lower().strip()

                                                                                                                                                                                                                                                                                                                                                                                                                                            return jsonify(correct=bool(correct))

                                                                                                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                print(f"/evaluate-semantic handler error: {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            traceback.print_exc()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return jsonify(error='internal server error'), 500

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            return app


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if __name__ == '__main__':
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # For compatibility with Hugging Face Spaces, bind to 0.0.0.0 and use $PORT
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    port = int(os.getenv('PORT', '7860'))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        app = create_app()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            app.run(host='0.0.0.0', port=port)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            