import re 

def build_triviaqa_prompt(item: dict):
    question = item["question"]
    
    system_message = (
        "You are a trivia expert. Please answer questions in exactly this format:\n"
        "Answer: [1-3 words only]\n"
        "Certainty: [0-100]\n\n"
    )
    
    user_prompt = f"Question: {question}"
    
    return system_message, user_prompt

def build_math_prompt(item: dict):
    question = item["problem"]

    system_message = (
        "You are a math expert. Please answer questions in exactly this format:\n"
        "Answer: [final answer only, e.g. 42, 3/5, sqrt(2)]\n"
        "Certainty: [0-100]\n\n"
    )

    user_prompt = f"Question: {question}"

    return system_message, user_prompt

def _last_boxed(solution: str):
    """The body of the last \\boxed{...} in a solution, matched brace by brace.

    A regex cannot do this: MATH answers nest braces (\\boxed{\\frac{1}{2}}), and the
    final answer is the last box, not the first.
    """
    idx = max(solution.rfind("\\boxed{"), solution.rfind("\\fbox{"))
    if idx < 0:
        return None
    start = solution.index("{", idx) + 1
    depth = 1
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i]
    return None                      # unbalanced braces


def extract_boxed_math(item: dict):
    """The reference answer of a MATH item, or None when the solution boxes nothing.

    Fractions, radicals, intervals and symbols are all legitimate MATH answers, so the
    latex is kept as written and only cleaned of the macros that carry no mathematical
    meaning. Callers are expected to drop the None items rather than score against them.
    """
    answer = _last_boxed(item["solution"])
    if answer is None:
        return None
    answer = re.sub(r"\\(?:text|mbox|textbf)\{([^{}]*)\}", r"\1", answer)   # \text{ cm}
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\\!|\\,|\\;|\\:|\\quad|\\qquad", "", answer)
    answer = re.sub(r"\\[dt]frac", r"\\frac", answer)                      # \dfrac -> \frac
    # \$ before $: dropping the dollar first leaves the backslash behind, which turned
    # \boxed{\$20} into "\20" -- an answer no prediction can ever match
    answer = answer.replace("\\$", "").replace("$", "").strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer or None

def build_gsm8k_prompt(item: dict):
    question = item["question"]
    
    system_message = (
        "Please answer questions in exactly this format:\n"
        "Answer: [number]\n"
        "Certainty: [0-100]\n\n"
    )
    
    user_prompt = f"Question: {question}"
    
    return system_message, user_prompt

def build_usmle_prompt(item: dict):
    
    system_message = (
        "You are a medical expert. Please answer questions in exactly this format:\n"
        "Answer: <one of [a, b, c, d]>\n" # Answer: <one of [a, b, c, d]> <2-5 words>" # 
        "Certainty: <between 0-100>\n\n"
    )
    user_prompt = (
        "Context:"
        + " ".join(item["question"].split(". ")[:-1])
        + "\n"
        + "Question:"
        + item["question"].split(". ")[-1]
        + "\n"
        + "Options:"
        + " ".join(str(v) for v in item["options"].items())
    )
    return system_message, user_prompt

def build_ind_prompt(item: dict):
    
    system_message = (
        "You are a medical expert. Please answer questions in exactly this format:\n"
        "Answer: [repeat correct option]\n" # Answer: <one of [a, b, c, d]> <2-5 words>" # 
        "Certainty: [0-100]\n\n"
    )
    user_prompt = (
        "Question: \n"
        + item["question"].split(". ")[-1]
        + "\n"
        + "Options: \n"
        + " ".join(str(item[op]) + "\n" for i, op in enumerate(["opa", "opb", "opc", "opd"]))
    )
    return system_message, user_prompt


def build_hotpot_prompt(item: dict):
    system_message = (
        "You are a helpful assistant."
        "Answer the question using the information in the provided passages."
        "Please answer questions in exactly this format:\n"
        "Answer: [1-5 words only]\n"
        "Certainty: [0-100]\n\n"
    )
    list_sentences = []
    for sentence_list in item["context"]["sentences"]:
        list_sentences.extend(sentence_list)

    context_text = "\n".join(sentence for sentence in list_sentences)

    user_prompt = (
        f"Question:\n"
        f"{item['question']}\n"
        "Context:\n"
        f"{context_text}\n\n"
    )

    return system_message, user_prompt


def build_el_prompt(item: dict):
    system_message = (
        "You are a helpful assistant. Please answer questions in exactly this format:\n"
        "Please answer questions in exactly this format:\n"
        "Answer: [1-5 sentences only]\n"
        "Certainty: [0-100]\n\n"
    )

    user_prompt = (
        f"Question:\n"
        f"{item['question']}\n"
    )

    return system_message, user_prompt

def build_fever_prompt(item: dict):
    claim = item["premise"]
    evidence = item.get("hypothesis", "")
    system_message = (
        "You are a fact-checking expert. Given a claim and evidence, decide whether the evidence supports the claim, refutes it, or contains not enough information. Please answer questions in exactly this format:\n"
        "Answer: [SUPPORTS | REFUTES | NOT ENOUGH INFO]\n"
        "Certainty: [0-100]\n\n"
    )
    user_prompt = f"Evidence: {evidence}\nClaim: {claim}"
    return system_message, user_prompt