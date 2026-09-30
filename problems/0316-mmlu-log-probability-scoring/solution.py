import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    preds = np.argmax(log_probs, axis=-1)
    probs = np.exp(log_probs - np.max(log_probs, axis=-1, keepdims=True))
    probs = probs/np.sum(probs, axis=-1, keepdims=True)

    top_lp = np.take_along_axis(probs, np.array(correct_answers)[..., None], axis=-1).squeeze(-1)

    return {"accuracy": round(np.mean(preds == np.array(correct_answers)), 4),
            "predictions": preds.tolist(),
            "avg_correct_prob": np.mean(top_lp)}