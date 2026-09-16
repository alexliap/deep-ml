def mmlu_letter_matching(model_outputs: list[str], ground_truth: list[str], subjects: list[str]) -> dict:
    """
    Evaluate MMLU predictions using letter-matching.
    
    Args:
        model_outputs: List of model generated responses
        ground_truth: List of correct answer letters (A, B, C, or D)
        subjects: List of subject names for each question
    
    Returns:
        Dictionary with evaluation metrics
    """
    import re
    import numpy

    pattern = re.compile(
        r'(?i)(?:\b(?:answer|option|choice)\s*(?:is|:)?\s*)?[\(\[]?\b([A-D])\b[\)\]]?'
    )

    result = {"overall_accuracy": 0,
        "subject_accuracy": {},
        "valid_response_rate": 0,
        "total_correct": 0,
        "total_questions": 0,
    }

    per_subj_perf = {sub: [] for sub in subjects}

    for i in range(len(model_outputs)):
        result["total_questions"] += 1
        # model model_output
        model_output = model_outputs[i]
        # ground_truth
        gt = ground_truth[i]
        # subject
        subject = subjects[i]

        matches = pattern.findall(model_output)
        match = matches[-1].upper() if matches else None

        if match:
            result["valid_response_rate"] += 1
            if match == gt:
                per_subj_perf[subject].append(1)
                result["total_correct"] += 1

    result["valid_response_rate"] = result["valid_response_rate"] / result["total_questions"]

    per_subj_perf = {sub: numpy.mean(per_subj_perf[sub]) for sub in subjects}

    result["subject_accuracy"] = per_subj_perf
    result["overall_accuracy"] = result["total_correct"] / result["total_questions"]



    return result


