param(
    [string]$ProjectRoot = ".",
    [string]$Step9File = "",
    [string]$OutDir = "",
    [string]$PythonExe = "python",
    [switch]$Run
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string]$PathText) {
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $PathText))
}

$ProjectRoot = Resolve-FullPath $ProjectRoot

if (-not (Test-Path $ProjectRoot -PathType Container)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $ProjectRoot "_work\step9_projected_boundary_flux"
}
$OutDir = Resolve-FullPath $OutDir

Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "OutDir      = $OutDir"

if (Test-Path $OutDir) {
    Remove-Item $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutDir | Out-Null

# Copy project to an isolated workspace. Original files are not modified.
$excludeDirs = @(".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "_work")
$excludeFiles = @("*.pyc", "*.pyo", "*.pyd", "*.log")

$robocopyArgs = @(
    $ProjectRoot,
    $OutDir,
    "/E",
    "/XD"
) + $excludeDirs + @(
    "/XF"
) + $excludeFiles + @(
    "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP"
)

$rc = (Start-Process -FilePath "robocopy" -ArgumentList $robocopyArgs -Wait -PassThru).ExitCode
if ($rc -ge 8) {
    throw "robocopy failed with exit code $rc"
}

if ([string]::IsNullOrWhiteSpace($Step9File)) {
    $candidates = Get-ChildItem -Path $OutDir -Recurse -File -Include *.py |
        Where-Object {
            $_.FullName -notmatch "\\(_work|\.git|__pycache__)\\"
        } |
        Where-Object {
            $_.Name -match "(?i)step\s*9|step9|convergence|conv"
        } |
        Sort-Object FullName

    if ($candidates.Count -eq 0) {
        throw "Cannot auto-detect Step9 convergence python file. Pass -Step9File explicitly."
    }

    $Step9Path = $candidates[0].FullName
} else {
    $candidateInCopy = Join-Path $OutDir ([IO.Path]::GetRelativePath($ProjectRoot, (Resolve-FullPath $Step9File)))
    if (Test-Path $candidateInCopy -PathType Leaf) {
        $Step9Path = $candidateInCopy
    } else {
        $Step9Full = Resolve-FullPath $Step9File
        if (-not (Test-Path $Step9Full -PathType Leaf)) {
            throw "Step9File not found: $Step9File"
        }
        $rel = [IO.Path]::GetFileName($Step9Full)
        $Step9Path = Join-Path $OutDir $rel
        Copy-Item $Step9Full $Step9Path -Force
    }
}

Write-Host "Step9 source in workspace = $Step9Path"

$dir = Split-Path $Step9Path -Parent
$base = [IO.Path]::GetFileNameWithoutExtension($Step9Path)
$ProjectedFile = Join-Path $dir ($base + "_projected_boundary_flux.py")
Copy-Item $Step9Path $ProjectedFile -Force

$patcher = @'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

helper = r"""
# === SDG projected-boundary-flux patch: BEGIN ===
# Strong-form projected-SBP consistency:
#   Psdg = V (V^T W V)^(-1) V^T W
#   q^-       -> E Psdg q^-
#   F_n^-     -> n_eta E Psdg F_eta^- + n_zeta E Psdg F_zeta^-
#   p         -> F_n,P^- - F_n^*
# Do NOT use Psdg on p after the penalty is formed.
def _sdg_make_projector(V, W):
    import numpy as _np
    Wm = _np.diag(W) if getattr(W, "ndim", 2) == 1 else W
    H = V.T @ Wm @ V
    return V @ _np.linalg.solve(H, V.T @ Wm)

def _sdg_ensure_projector(_locals, _globals):
    if "Psdg" in _locals:
        return _locals["Psdg"]
    if "Psdg" in _globals:
        return _globals["Psdg"]

    V_ = _locals.get("V", _globals.get("V", None))
    W_ = _locals.get("W", _globals.get("W", None))

    if V_ is None:
        V_ = _locals.get("Vq", _globals.get("Vq", None))
    if W_ is None:
        W_ = _locals.get("wq", _globals.get("wq", None))
    if W_ is None:
        W_ = _locals.get("Wq", _globals.get("Wq", None))

    if V_ is None or W_ is None:
        raise NameError("Cannot build Psdg: expected V/W or Vq/wq in the current scope.")

    P_ = _sdg_make_projector(V_, W_)
    _locals["Psdg"] = P_
    return P_

def _sdg_trace(E, Psdg, x):
    return E @ (Psdg @ x)

def _sdg_projected_normal_flux(E, Psdg, Feta, Fzeta, n_eta, n_zeta):
    return n_eta * (E @ (Psdg @ Feta)) + n_zeta * (E @ (Psdg @ Fzeta))
# === SDG projected-boundary-flux patch: END ===
"""

if "_sdg_make_projector" not in text:
    m = list(re.finditer(r"^(?:from\s+\S+\s+import\s+.*|import\s+.*)$", text, flags=re.M))
    insert_at = m[-1].end() if m else 0
    text = text[:insert_at] + "\n" + helper + "\n" + text[insert_at:]

# Conservative replacements. These are intentionally narrow:
#   qM = E @ q           -> qM = E @ (Psdg @ q)
#   fnM = n1*(E@F1)+...  -> fnM = n1*(E@(Psdg@F1))+...
patch_count = 0

def insert_projector_guard(before_line: str) -> str:
    indent = re.match(r"^\s*", before_line).group(0)
    return (
        f"{indent}Psdg = _sdg_ensure_projector(locals(), globals())\n"
        + before_line
    )

lines = text.splitlines(keepends=True)
new_lines = []

normal_flux_regexes = [
    # fn = n_eta * (E @ Feta) + n_zeta * (E @ Fzeta)
    re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z_]\w*)\s*=\s*"
        r"(?P<neta>[A-Za-z_]\w*)\s*\*\s*\(\s*(?P<E>[A-Za-z_]\w*)\s*@\s*(?P<Feta>[A-Za-z_]\w*)\s*\)\s*\+\s*"
        r"(?P<nzeta>[A-Za-z_]\w*)\s*\*\s*\(\s*(?P=E)\s*@\s*(?P<Fzeta>[A-Za-z_]\w*)\s*\)\s*(?P<comment>#.*)?$"
    ),
    # fn = (n_eta * (E @ Feta)) + (n_zeta * (E @ Fzeta))
    re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z_]\w*)\s*=\s*"
        r"\(\s*(?P<neta>[A-Za-z_]\w*)\s*\*\s*\(\s*(?P<E>[A-Za-z_]\w*)\s*@\s*(?P<Feta>[A-Za-z_]\w*)\s*\)\s*\)\s*\+\s*"
        r"\(\s*(?P<nzeta>[A-Za-z_]\w*)\s*\*\s*\(\s*(?P=E)\s*@\s*(?P<Fzeta>[A-Za-z_]\w*)\s*\)\s*\)\s*(?P<comment>#.*)?$"
    ),
]

trace_regexes = [
    # qM = E @ q
    re.compile(
        r"^(?P<indent>\s*)(?P<lhs>q(?:M|m|Minus|minus|_minus|_M|_m|L|l|Left|left)?|q_face|qf|qb)\s*=\s*"
        r"(?P<E>E|E0|Ef|Eface|E_face)\s*@\s*(?P<q>q|Q|q0|q_local|q_old|u)\s*(?P<comment>#.*)?$"
    )
]

for line in lines:
    stripped = line.strip()

    if "Psdg @ p" in line or "Psdg@p" in line:
        raise RuntimeError("Unsafe pattern detected: projector is being applied to p. This patch refuses to continue.")

    replaced = False

    for rgx in normal_flux_regexes:
        mm = rgx.match(line.rstrip("\n"))
        if mm:
            g = mm.groupdict()
            repl = (
                f"{g['indent']}Psdg = _sdg_ensure_projector(locals(), globals())\n"
                f"{g['indent']}{g['lhs']} = {g['neta']} * ({g['E']} @ (Psdg @ {g['Feta']})) "
                f"+ {g['nzeta']} * ({g['E']} @ (Psdg @ {g['Fzeta']}))"
                f"{' ' + g.get('comment') if g.get('comment') else ''}\n"
            )
            new_lines.append(repl)
            patch_count += 1
            replaced = True
            break

    if replaced:
        continue

    for rgx in trace_regexes:
        mm = rgx.match(line.rstrip("\n"))
        if mm:
            g = mm.groupdict()
            repl = (
                f"{g['indent']}Psdg = _sdg_ensure_projector(locals(), globals())\n"
                f"{g['indent']}{g['lhs']} = {g['E']} @ (Psdg @ {g['q']})"
                f"{' ' + g.get('comment') if g.get('comment') else ''}\n"
            )
            new_lines.append(repl)
            patch_count += 1
            replaced = True
            break

    if not replaced:
        new_lines.append(line)

text = "".join(new_lines)

# Add a visible sentinel near penalty definitions, but do not alter p itself.
text = re.sub(
    r"(^\s*p\s*=\s*.*$)",
    r"\1  # SDG projected patch: p uses projected interior flux/state; do not project p",
    text,
    count=1,
    flags=re.M
)

if patch_count == 0:
    raise RuntimeError(
        "No boundary trace / normal-flux pattern was patched. "
        "Open the copied file and replace raw E@q, E@Feta, E@Fzeta manually by E@(Psdg@...)."
    )

path.write_text(text, encoding="utf-8")
print(f"patched_file={path}")
print(f"patch_count={patch_count}")
'@

$patcherPath = Join-Path $OutDir "__patch_projected_boundary_flux.py"
Set-Content -Path $patcherPath -Value $patcher -Encoding UTF8

& $PythonExe $patcherPath $ProjectedFile
if ($LASTEXITCODE -ne 0) {
    throw "Python patcher failed."
}

Write-Host ""
Write-Host "Created projected-boundary-flux copy:"
Write-Host "  $ProjectedFile"
Write-Host ""
Write-Host "Important implemented rule:"
Write-Host "  q^-   := E Psdg q^-"
Write-Host "  F_n^- := n_eta E Psdg F_eta^- + n_zeta E Psdg F_zeta^-"
Write-Host "  p     := F_n,P^- - F_n^*"
Write-Host "  Do not apply Psdg to p."
Write-Host ""

if ($Run) {
    Push-Location $dir
    try {
        & $PythonExe $ProjectedFile
        if ($LASTEXITCODE -ne 0) {
            throw "Projected Step9 run failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
