#!/bin/bash
# Run every misuse case of test_must_fail and check that each one stops with
# a nonzero exit AND prints the expected message.
# Usage: test/must_fail.sh <path to test_must_fail>
exe="$1"
cases=(
   "copy_scalar|assignment of dp_2d_t is not allowed (copying 'original')"
   "copy_array|assignment of dp_1d_t is not allowed"
   "realloc_method|'x' was reallocated or deallocated behind devarray"
   "realloc_collective|'x' was reallocated or deallocated behind its owner"
   "unset_tag|invalid mem_tag_t"
   "unknown_tag|no tag named 'swe_sate'"
   "stale_tag|invalid mem_tag_t"
   "long_tag|longer than MEM_TAG_NAME_LEN"
   "bounds|'a' has bounds (0:4, 0:4), expected (1:4, 1:4)"
   "double_map|'x' is already on the device"
   "reset_live|objects are still live"
)
fails=0
for c in "${cases[@]}"; do
   name="${c%%|*}"
   want="${c#*|}"
   out="$("$exe" "$name" 2>&1)"
   rc=$?
   if [ $rc -ne 0 ] && grep -qF -- "$want" <<<"$out"; then
      echo "  ok   must fail: $name"
   else
      echo "  FAIL must fail: $name (exit $rc)"
      echo "$out" | sed 's/^/         /'
      fails=$((fails + 1))
   fi
done
if [ $fails -ne 0 ]; then
   echo "test_must_fail: $fails case(s) FAILED"
   exit 1
fi
echo "test_must_fail: all ${#cases[@]} misuse cases stopped with the right message"
