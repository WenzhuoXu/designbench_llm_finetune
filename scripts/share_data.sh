#!/bin/bash
# Relocate the generated data into the project's shared folder, leaving symlinks
# behind so existing paths keep working.
#
# mv, not cp: /ocean/projects/mch250030p/{wxu7,shared} are the same Lustre
# device, so this is a metadata rename -- instant, and it does not spend the
# project's 405 GB quota twice. The quota is pooled across every user in
# mch250030p, so a copy would cost the group 3.6 GB for nothing.
set -euo pipefail

P=/ocean/projects/mch250030p
DEST=$P/shared/designbench
LF=$P/wxu7/llm_finetune
DB=$P/wxu7/DesignBench

mkdir -p "$DEST/problems"
# setgid so anything created below inherits the project group rather than the
# creator's primary group.
chmod g+rwxs "$DEST" "$DEST/problems"

relocate() {           # relocate <source> <destination>
  local src="$1" dst="$2"
  if [ -L "$src" ]; then
    echo "  already a symlink, skipping: $src"
    return
  fi
  if [ ! -e "$src" ]; then
    echo "  missing, skipping: $src"
    return
  fi
  if [ -e "$dst" ]; then
    echo "  destination exists, skipping: $dst"
    return
  fi
  mv "$src" "$dst"
  ln -s "$dst" "$src"
  echo "  moved $(basename "$src") -> $dst"
}

echo "corpus and tiers:"
relocate "$LF/data/corpus_v2" "$DEST/corpus_v2"
relocate "$LF/data/tiers"     "$DEST/tiers"

echo "problem sets:"
relocate "$DB/data/problems_hard" "$DEST/problems/problems_hard"
for k in c d e f g; do
  relocate "$DB/data/problems_gen_$k" "$DEST/problems/problems_gen_$k"
done

echo "setting group access (read + traverse for the project group)"
chmod -R g+rX "$DEST"
find "$DEST" -type d -exec chmod g+ws {} + 2>/dev/null || true

echo
echo "result:"
du -sh "$DEST" 2>/dev/null
ls -l "$LF/data" | grep -E "corpus_v2|tiers" || true
