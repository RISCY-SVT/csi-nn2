#!/usr/bin/env bash
# run_rvv_tests.sh — запуск *.elf-тестов с цветным выводом и итоговым отчётом.
# Usage: ./run_rvv_tests.sh [/path/to/elf/dir]

set -uo pipefail              # «-e» убрали!

DIR=${1:-/tmp}
shopt -s nullglob             # если файлов нет, for получит пустой список

# ─── цвета ────────────────────────────────────────────────────────────────
RED=$'\e[31m'
GREEN=$'\e[32m'
NC=$'\e[0m'

declare -a failed passed

echo -e "Running RVV unit-tests in directory: ${DIR}\n"

for exe in "${DIR}"/*.elf; do
  [[ -x ${exe} ]] || continue

  echo "▶ ${exe}"

  # ── запускаем тест, отключив «-e», чтобы не оборвать скрипт ──
  set +e
  output=$("${exe}" 2>&1)
  exit_code=$?
  set -e                    # включаем обратно (не обязательно, но можно)

  # ── критерии провала ──
  if (( exit_code != 0 )) || \
     grep -q -E 'Failed|Segmentation fault' <<<"${output}"; then
      echo -e "${RED}❌  FAILED${NC}"
      failed+=("${exe}")
  else
      echo -e "${GREEN}✅  OK${NC}"
      passed+=("${exe}")
  fi

  echo "${output}"
  echo
done

# ─── Summary ──────────────────────────────────────────────────────────────
total=$(( ${#passed[@]} + ${#failed[@]} ))

echo "──────────  SUMMARY  ──────────"
echo -e "Total tests: ${total}"
echo -e "${GREEN}Passed: ${#passed[@]}${NC}"
echo -e "${RED}Failed: ${#failed[@]}${NC}"

if ((${#passed[@]})); then
  echo -e "\nPassed list:"
  for p in "${passed[@]}"; do
    echo -e "  ${GREEN}${p}${NC}"
  done
fi

if ((${#failed[@]})); then
  echo -e "\nFailed list:"
  for f in "${failed[@]}"; do
    echo -e "  ${RED}${f}${NC}"
  done
fi
echo "────────────────────────────────"
