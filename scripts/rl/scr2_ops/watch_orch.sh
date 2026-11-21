#!/bin/bash
# watch_orch.sh <until 'YYYY-MM-DD HH:MM'> [period s=1200]: every period, the Orchard health line; ends (so the session is
# told) when the health check asks for a look, the family changes, the sampler finishes, the link fails three times
# running, or the time is reached.
S="C:/Users/xuwen/AppData/Local/Temp/claude/6677f76fa28a0973/9311749f-16a4-40d4-a1a2-48798138b96f/scratchpad"
END=$(date -d "$1" +%s); P="${2:-1200}"; FAIL=0
while :; do
  OUTP=$(bash "$S/orr.sh" ssh "bash ~/rl/logs/fcswap/health.sh; echo HEALTH_RC=\$?" 2>&1)
  RC=$(echo "$OUTP" | sed -n 's/^HEALTH_RC=//p' | tail -n 1)
  echo "$OUTP" | grep -v '^HEALTH_RC=' | tail -n 3
  if [ -z "$RC" ]; then
    FAIL=$((FAIL+1)); echo "$(date '+%H:%M') no answer from Orchard ($FAIL)"
    [ "$FAIL" -ge 3 ] && { echo "WATCH END: the link to Orchard fails"; exit 3; }
  else
    FAIL=0
    [ "$RC" = 0 ] || { echo "WATCH END: health rc=$RC"; exit 0; }
  fi
  [ "$(date +%s)" -ge "$END" ] && { echo "WATCH END: time reached"; exit 0; }
  sleep "$P"
done
