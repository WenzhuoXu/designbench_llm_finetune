import pathlib
p = pathlib.Path("/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_run.py")
s = p.read_text()
s = s.replace('for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",\n          "/ocean/projects/mch250030p/wxu7/DesignBench"):',
              'for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",\n          "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts",\n          "/ocean/projects/mch250030p/wxu7/DesignBench"):')
s = s.replace('    except Exception as e:\n        return spec["problem_id"], "planner", None, 0, 0',
              '    except Exception as e:\n        print("PLANNER FAIL:", type(e).__name__, str(e)[:160], flush=True)\n        return spec["problem_id"], "planner", None, 0, 0')
p.write_text(s)
print("patched; scripts on path:", "scripts" in s)
