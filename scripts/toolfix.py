"""Repair the two topology tools, which have been failing silently since v1.

DesignBench's executor expects:
    ADD_MEMBER(j1, j2, material, Pipe(r=..., t=...))      -- shape with parenthesised params
    MOVE_JOINT(id, [x, y, z])                             -- three coordinates
The tool layer was emitting ADD_MEMBER with comma-separated shape params, which never matched
the executor's regex, and MOVE_JOINT with two coordinates. Both returned None and were counted
as "proposal could not be applied", so the planner's only two topology moves were dead.

This rewrites those two branches of apply_tool in place and leaves the sizing tools alone.
"""
import io, re
p = "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/planner01.py"
s = io.open(p, encoding="utf-8").read()

old_add = '''    if name == "ADD_MEMBER" and len(nums) >= 2:
        return H.apply_candidate(cur, "ADD_MEMBER(%d, %d, 6061_T6_Aluminum, Pipe, 0.030, 0.004)"
                                 % (int(float(nums[0])), int(float(nums[1])))) or cur'''
new_add = '''    if name == "ADD_MEMBER" and len(nums) >= 2:
        # the executor wants the shape with parenthesised params, not comma-separated
        return H.apply_candidate(
            cur, "ADD_MEMBER(%d, %d, 6061_T6_Aluminum, Pipe(r=0.030, t=0.004))"
            % (int(float(nums[0])), int(float(nums[1])))) or cur'''
assert old_add in s, "ADD_MEMBER branch not found"
s = s.replace(old_add, new_add)

old_move = '''    if name == "MOVE_JOINT" and len(nums) >= 3:
        j = int(float(nums[0]))
        return H.apply_candidate(cur, "MOVE_JOINT(%d, [0.0, 0.0], [%.4f, %.4f])"
                                 % (j, float(nums[1]), float(nums[2]))) or cur'''
new_move = '''    if name == "MOVE_JOINT" and len(nums) >= 3:
        # three coordinates; the executor ignores any old position and uses the new one
        j = int(float(nums[0]))
        z = float(nums[3]) if len(nums) >= 4 else 0.0
        return H.apply_candidate(cur, "MOVE_JOINT(%d, [%.4f, %.4f, %.4f])"
                                 % (j, float(nums[1]), float(nums[2]), z)) or cur'''
assert old_move in s, "MOVE_JOINT branch not found"
s = s.replace(old_move, new_move)

io.open(p, "w", encoding="utf-8").write(s)
print("patched apply_tool: ADD_MEMBER and MOVE_JOINT")
