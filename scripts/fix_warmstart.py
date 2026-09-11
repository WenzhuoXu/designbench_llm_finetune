"""
Let the training pipeline keep whole expert moves.

The expert traces are byte-correct and replay to feasible, but the warmstart target damages
1,107 of their action turns before the model ever sees them. Two independent causes:

  - WarmstartTransform._extract_action calls grammar.extract_action, which returns the FIRST
    valid action. 927 turns are compounds joined by this repo's own ' ; ' separator, averaging
    11 parts, so the model would be trained to emit one member's rescale where the expert
    rescaled the whole structure. find_actions already returns every part; only the choice to
    keep one was wrong.

  - grammar._validate_move_joint requires 2-element coordinates, while trussme's executor
    requires 3-element ones, so the 63 MOVE_JOINT turns validate as invalid and are emptied.
    The executor is the authority on what runs; the validator should accept what it accepts.

Both patched in place, minimally, with single-action behaviour unchanged.
"""
import io

# ---- 1. keep every part of a compound action -------------------------------------------
p = "/ocean/projects/mch250030p/wxu7/llm_finetune/llm_finetune/data/processors/warmstart_transform.py"
s = io.open(p, encoding="utf-8").read()

old = '''    def _extract_action(self, think_content: str) -> Optional[str]:
        """Extract grammar action line from shallow <think> content."""
        return extract_action(think_content)'''

new = '''    def _extract_action(self, think_content: str) -> Optional[str]:
        """Extract the grammar action from shallow <think> content.

        Expert trajectories carry compound moves joined by ' ; ' -- a fully-stressed pass is
        one rescale per member, eleven parts on average. ``extract_action`` returns only the
        first valid one, which would train the model to make a fraction of the move the
        expert made. Every valid part is kept and rejoined; a single-action turn is
        unaffected.
        """
        from llm_finetune.data.grammar import find_actions, validate_action

        parts = []
        for candidate in find_actions(think_content):
            result = validate_action(candidate)
            if result.is_valid and result.action not in parts:
                parts.append(result.action)
        if not parts:
            return extract_action(think_content)
        return " ; ".join(parts)'''

assert old in s, "warmstart _extract_action not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched warmstart_transform._extract_action")

# ---- 2. accept the coordinate arity the executor actually requires ----------------------
g = "/ocean/projects/mch250030p/wxu7/llm_finetune/llm_finetune/data/grammar.py"
s = io.open(g, encoding="utf-8").read()

old2 = '''    if _parse_float_list(args[1], expected_len=2) is None:
        return "old_position_invalid"
    if _parse_float_list(args[2], expected_len=2) is None:
        return "new_position_invalid"
    return ""'''

new2 = '''    # trussme's executor takes 3-vectors; this validator demanded 2 and so rejected every
    # MOVE_JOINT the executor accepts. Accept either, since the executor is the authority.
    if (_parse_float_list(args[1], expected_len=2) is None
            and _parse_float_list(args[1], expected_len=3) is None):
        return "old_position_invalid"
    if (_parse_float_list(args[2], expected_len=2) is None
            and _parse_float_list(args[2], expected_len=3) is None):
        return "new_position_invalid"
    return ""'''

assert old2 in s, "_validate_move_joint body not found"
io.open(g, "w", encoding="utf-8").write(s.replace(old2, new2, 1))
print("patched grammar._validate_move_joint")
