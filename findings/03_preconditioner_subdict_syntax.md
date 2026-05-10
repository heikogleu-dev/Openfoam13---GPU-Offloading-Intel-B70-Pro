# Bug: OGL Preconditioner Options Silently Ignored with Scalar Syntax

## Symptom

`maxBlockSize` (and other preconditioner options like `skipSorting`,
`caching`, `multiLevelSchwarz`) are **silently ignored** when using the
scalar `preconditioner` syntax.

The verbose log shows the default value:
```
[OGL LOG][Preconditioner.hpp:176] Generate preconditioner BJ<double> MaxBlockSize 1
```

…even though fvSolution had `maxBlockSize 32`.

## Wrong Syntax (Options Ignored)

```
p
{
    solver           GKOCG;
    preconditioner   BJ;     ← scalar value
    maxBlockSize     32;     ← IGNORED (d == dictionary::null)
    skipSorting      true;   ← IGNORED
    caching          1;      ← IGNORED
}
```

## Correct Syntax (Sub-Dictionary Required)

```
p
{
    solver           GKOCG;
    preconditioner          ← sub-dictionary key
    {
        preconditioner  BJ;     ← name as sub-key
        maxBlockSize    32;     ← these now work
        skipSorting     true;
        multiLevelSchwarz false;
        caching         1;
    }
}
```

## Root Cause

OGL's `Preconditioner::init_preconditioner()` (Preconditioner.hpp ~line 569):

```cpp
const entry &e = solverControls_.lookupEntry("preconditioner", true, true);
if (e.isDict()) {
    name = e.dict().lookup<word>("preconditioner");
} else {
    e.stream() >> name;
}
const dictionary &d = e.isDict() ? e.dict() : dictionary::null;
init_preconditioner_impl(name, d, ...);
// → d.lookupOrDefault("maxBlockSize", 1) returns 1 if d == dictionary::null
```

When `preconditioner BJ;` is a scalar entry, `e.isDict()` is `false` and
`d` becomes `dictionary::null` (empty dict). All option lookups inside
`init_preconditioner_impl` then return their defaults.

## Documentation Gap

The OGL README only shows the **scalar** syntax for the preconditioner
field. The sub-dictionary syntax — required to actually configure any
preconditioner option — is not documented in the README.

## Verification

After fixing the syntax, the verbose log shows the correct value:
```
[OGL LOG][Preconditioner.hpp:176] Generate preconditioner BJ<double> MaxBlockSize 32
```

…unfortunately, only to crash with OOM (see `02_bj_maxblocksize_oom.md` —
note: this finding was not published; see [02_bj_blocksize_int_underflow.md](02_bj_blocksize_int_underflow.md) for the related published finding) since the SYCL BJ-generate path can't allocate the workspace.
