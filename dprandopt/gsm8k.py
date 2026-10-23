"""Numeric normalization shared by GSM8K selection rewards and voting."""
from decimal import Decimal, InvalidOperation


def canonical_number(value):
    text=str(value).strip().replace(',', '').rstrip('.')
    try:
        number=Decimal(text)
    except InvalidOperation:
        return ''
    if not number.is_finite():
        return ''
    return '0' if number==0 else format(number.normalize(),'f')


def answer_and_reward(handler, text, ground_truth):
    # Use exactly the voting extractor: strict if present, otherwise flexible.
    # Do not retry flexible extraction after a wrong strict answer.
    answer=canonical_number(handler.extract_answer(text))
    gold=canonical_number(ground_truth)
    if not gold:
        raise ValueError('GSM8K ground truth must be numeric')
    return answer, float(bool(answer) and answer==gold)
