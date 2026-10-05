import numpy as np

from dprandopt import countdown as cd


def test_rewards_range_and_values():
    nums, target = [1, 2, 3], 9
    perfect = " (1+2) </think>\n<answer> (1 + 2) * 3 </answer>"
    assert cd.randopt_reward(perfect, nums, target) == 1.1
    bare = "<answer> (1 + 2) * 3 </answer>"
    assert cd.randopt_reward(bare, nums, target) == 1.0 + 0.1 * 0.5
    wrong = "<answer> 1 + 2 + 3 </answer>"
    assert cd.answer_reward(wrong, nums, target) == 0.0
    reuse = "<answer> 3 * 3 </answer>"
    assert cd.answer_reward(reuse, nums, target) == 0.0
    assert cd.answer_reward("<answer> 9 = (1 + 2) * 3 </answer>", nums, target) == 1.0


def test_voting_answer_values():
    assert cd.voting_answer("<answer>(1 + 2) * 3</answer>", [1, 2, 3]) == "9"
    assert cd.voting_answer("<answer>1 / 2 + 3</answer>", [1, 2, 3]) == "3.5"
    assert cd.voting_answer("<answer>1 + 2</answer>", [1, 2, 3]) == ""


def test_generated_instances_are_solvable():
    for inst in cd.generate_instances(40, seed=0):
        sol = cd.solve(inst["numbers"], inst["target"])
        assert sol is not None
        assert cd.answer_reward(f"<answer>{sol}</answer>", inst["numbers"], inst["target"]) == 1.0


def test_encode_answers():
    codes = cd.encode_answers([["9", ""], ["8", "5"], ["8", "4"]], ["9", "4"])
    assert codes.tolist() == [[0, -1], [1, 1], [1, 0]]
