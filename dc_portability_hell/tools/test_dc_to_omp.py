"""Structural regression guards for dc_to_omp.py's block finder.

Every test here corresponds to a bug found by running the translator over
MOM6, whose `;`-compound statement style ("endif ; enddo",
"if (m==1) then ; do concurrent (...) ; press(i,j) = 0.0 ; enddo") breaks
line-anchored parsing in ways that are silent rather than loud.
"""
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(_HERE))
    spec.loader.exec_module(mod)
    return mod


d = _load("dc_to_omp")


def _xform(src):
    lines, n = d.transform_lines(src.splitlines(), "gpu")
    return "\n".join(lines), n


def test_mask_does_not_abandon_rest_of_file():
    """A skipped loop must not end the scan — None means 'no more loops'."""
    src = (
        "do concurrent (i=1:n, mask(i))\n"
        "  a(i) = 1\n"
        "end do\n"
        "do concurrent (j=1:m)\n"
        "  b(j) = 2\n"
        "end do\n")
    out, n = _xform(src)
    assert n == 1, "the loop after the masked one must still convert"
    assert "do concurrent (i=1:n, mask(i))" in out, "masked loop left alone"
    assert "!$omp target teams distribute parallel do" in out


def test_string_literal_is_not_a_loop():
    """A character literal naming the construct must not derail the scan."""
    src = (
        'call check(error, ok, "do concurrent reduce(+) must be exact")\n'
        "do concurrent (i=1:n)\n"
        "  a(i) = 1\n"
        "end do\n")
    out, n = _xform(src)
    assert n == 1
    assert 'call check(error, ok, "do concurrent reduce(+) must be exact")' in out


def test_compound_enddo_closes_depth():
    """`endif ; enddo` closes an inner do; the outer end do is the match."""
    lines = [
        "do concurrent (i=1:n)",          # 0  header
        "  do k=1,nk ; if (c(k)) then",   # 1  opens a do
        "    x = 1",                      # 2
        "  endif ; enddo",                # 3  closes it — NOT at line start
        "end do",                         # 4  the true match
    ]
    assert d.find_matching_end_do(lines, 0) == 4


def test_compound_do_is_counted():
    """A `do` after `;` must open depth, or an outer end do is stolen."""
    lines = [
        "do concurrent (i=1:n)",              # 0  header
        "  if (m==1) then ; do k=1,nk",       # 1  opens a do, not at line start
        "    x = 1",                          # 2
        "  enddo ; endif",                    # 3  closes it
        "end do",                             # 4  the true match
    ]
    # The old line-anchored matcher missed the `do` on line 1 and returned 3.
    assert d.find_matching_end_do(lines, 0) == 4


def test_header_with_trailing_statements_refused():
    """MOM6's `do concurrent (...) ; if (...) then` must not be converted.

    The emitter replaces the whole header line, so converting would delete the
    trailing statement.
    """
    src = (
        "do concurrent (j=js:je, I=is-1:ie) ; if (mask(I,j) > 0.0) then\n"
        "  a(I,j) = 1\n"
        "endif ; enddo\n")
    out, n = _xform(src)
    assert n == 0
    assert "if (mask(I,j) > 0.0) then" in out


def test_one_line_loop_refused():
    """`do concurrent (i=is:ie) ; press(i,j) = 0.0 ; enddo` — leave it alone."""
    src = "do concurrent (i=is:ie) ; press(i,j) = 0.0 ; enddo\n"
    out, n = _xform(src)
    assert n == 0
    assert out.strip() == src.strip()


def test_shared_closing_line_refused():
    """A closing `end do` sharing its line must not be swallowed by the footer."""
    lines = [
        "do concurrent (i=1:n)",
        "  x = 1",
        "end do ; y = 2",
    ]
    assert d.find_matching_end_do(lines, 0) is None


def test_skip_message_reports_original_line():
    """Line numbers must survive the drift from preceding rewrites."""
    import io
    import contextlib
    src = (
        "do concurrent (i=1:n)\n"      # 1  converts: 1 line -> 3
        "  a(i) = 1\n"                 # 2
        "end do\n"                     # 3
        "do concurrent (j=1:m, msk(j))\n"  # 4  skipped — must report line 4
        "  b(j) = 2\n"
        "end do\n")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        _xform(src)
    assert "line 4" in err.getvalue(), err.getvalue()


def test_locality_macro_survives_to_private():
    src = (
        "do concurrent (j=1:ny, i=1:nx) DO_LOCALITY(local(tmp))\n"
        "  tmp = a(i,j)\n"
        "  b(i,j) = tmp\n"
        "end do\n")
    out, n = _xform(src)
    assert n == 1
    assert "private(tmp)" in out
